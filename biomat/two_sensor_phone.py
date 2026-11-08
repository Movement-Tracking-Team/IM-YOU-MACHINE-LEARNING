"""Exploratory phone inference with the separately retrained two-sensor models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import torch

from .two_sensor import OUT, ROOT, HZ, LENGTH, SITES, PREPROCESS, canonicalize, new_model, predict, load
from .prepare import TRAIN, TARGETS
from .classifier import CLASSES
from .train import write_json


def read_phone(path):
    df = pd.read_csv(path)
    columns = ['common_timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    values = df[columns].to_numpy(dtype=np.float64)
    if len(values) < LENGTH or not np.isfinite(values).all():
        raise ValueError('Missing/invalid phone samples')
    if set(df.accel_units) != {'m/s^2'} or set(df.gyro_units) != {'deg/s'}:
        raise ValueError('Unexpected phone units')
    t, x = values[:, 0], values[:, 1:]
    dt = np.diff(t)
    rate = (len(t)-1)/(t[-1]-t[0])
    if (dt <= 0).any() or dt.max() > .05 or not 55 <= rate <= 65:
        raise ValueError('This adapter requires a continuous approximately 60 Hz recording')
    x[:, 3:] = np.deg2rad(x[:, 3:])
    return t, x, rate, df


def prepare_phone(session):
    # Phone 1 is thigh; phone 2 is shank. Model ordering is shank then thigh.
    loaded = [read_phone(session / name) for name in ['phone_2.csv', 'phone_1.csv']]
    origin = float(json.loads((session/'metadata.json').read_text())['start'])
    start, stop = max(a[0][0] for a in loaded), min(a[0][-1] for a in loaded)
    timeline = start + np.arange(int(np.floor((stop-start)*HZ))+1)/HZ
    if len(timeline) < LENGTH:
        raise ValueError('Insufficient synchronized overlap')
    x = np.concatenate([np.column_stack([np.interp(timeline,t,v[:,j]) for j in range(6)])
                        for t,v,_,_ in loaded],axis=1)
    x = sosfiltfilt(butter(4,20,fs=HZ,output='sos'),x,axis=0)
    starts = np.arange(0, len(x)-LENGTH+1, 30)
    windows = np.stack([x[s:s+LENGTH] for s in starts])
    canonical, quality = canonicalize(windows)
    # Fixed acquisition/low-motion gate, not a learned standing class or OOD detector.
    moving = np.ones(len(windows), dtype=bool)
    for site in range(2):
        gyro = windows[:,:,site*6+3:site*6+6]
        fraction = (np.linalg.norm(gyro,axis=2) > .25).mean(1)
        moving &= fraction >= .75
        moving &= quality[site]['principal_axis_fraction'] >= .6
    info = {'rates_hz': {site: v[2] for site,v in zip(SITES,loaded)},
            'input_sha256': {name:hashlib.sha256((session/name).read_bytes()).hexdigest()
                             for name in ['phone_1.csv','phone_2.csv']},
            'session_origin_host_s':origin, 'gating': 'Both phones >=75% samples angular speed >0.25 rad/s; transverse gyro principal-axis fraction >=0.6',
            'quality': [{key:val.tolist() for key,val in q.items()} for q in quality]}
    return canonical, starts, timeline-origin, moving, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    session = args.session.resolve()
    if not (OUT/'results.json').exists():
        raise RuntimeError('Complete the Camargo experiment before phone inference')
    torch.set_num_threads(4)
    x, starts, timeline, usable, info = prepare_phone(session)
    output = session/'analysis/two_sensor_camargo_v1'
    output.mkdir(parents=True,exist_ok=True)
    p, checkpoints, diagnostics = {}, {}, {}
    training = load(TRAIN)[0]
    for task in ['classifier','regressor']:
        checkpoint_path = OUT/'final'/task/'model.pt'
        saved = torch.load(checkpoint_path,weights_only=True,map_location='cpu')
        if saved['preprocess'] != PREPROCESS or saved['sites'] != SITES:
            raise ValueError('Checkpoint preprocessing differs from phone adapter')
        model = new_model(task)
        model.load_state_dict(saved['state_dict'],strict=True)
        mean,scale = saved['mean'].numpy(),saved['scale'].numpy()
        p[task] = predict(model,x,mean,scale,'cpu',task)
        if not np.isfinite(p[task]).all():
            raise ValueError('Nonfinite model predictions')
        np.savez(output/(task+'_window_predictions.npz'),predictions=p[task],usable=usable,
                 start_session_s=timeline[starts])
        checkpoints[task] = {'path':str(checkpoint_path.relative_to(ROOT)),
                             'sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
        train_energy = np.mean(((training-mean)/scale)**2,axis=(1,2))
        phone_energy = np.mean(((x-mean)/scale)**2,axis=(1,2))
        diagnostics[task] = {'training_energy_99th_percentile':float(np.quantile(train_energy,.99)),
                             'phone_window_energy':phone_energy.tolist(),
                             'caution': 'Simple input-scale diagnostic; not a calibrated domain-shift or unknown-activity detector.'}
    labels = p['classifier'].argmax(1)
    table = pd.DataFrame({'start_session_s':timeline[starts], 'end_session_s':timeline[starts]+(LENGTH-1)/HZ,
                          'usable_motion_window':usable,
                          'candidate_activity':[CLASSES[i] if ok else 'not_evaluated_low_motion_or_unstable_axis'
                                                for i,ok in zip(labels,usable)],
                          'uncalibrated_model_score':np.where(usable,p['classifier'].max(1),np.nan)})
    for i,c in enumerate(CLASSES):
        table[c+'_score'] = np.where(usable,p['classifier'][:,i],np.nan)
    table.to_csv(output/'activity_windows.csv',index=False)
    sums = np.zeros((len(timeline),3)); counts=np.zeros(len(timeline),dtype=int)
    lows=np.full_like(sums,np.inf); highs=np.full_like(sums,-np.inf)
    for k,start in enumerate(starts):
        if not usable[k]:
            continue
        values=p['regressor'][k]
        sums[start:start+LENGTH] += values
        counts[start:start+LENGTH] += 1
        lows[start:start+LENGTH] = np.minimum(lows[start:start+LENGTH],values)
        highs[start:start+LENGTH] = np.maximum(highs[start:start+LENGTH],values)
    avg=np.full_like(sums,np.nan)
    np.divide(sums,counts[:,None],out=avg,where=counts[:,None]>0)
    lows[counts==0]=np.nan; highs[counts==0]=np.nan
    angles=pd.DataFrame({'session_time_s':timeline,'contributing_windows':counts})
    for j,name in enumerate(TARGETS):
        angles[name+'_degrees_unvalidated']=avg[:,j]
    angles.to_csv(output/'joint_angles_experimental.csv',index=False)
    summary={'session':session.name,'models':checkpoints,'preprocessing':PREPROCESS,
             'phone_accuracy_validated':False,'joint_angle_error_measurable':False,
             'user_confirmed_mounting':'Right thigh phone_1 and right shank phone_2, front mounted, screens out, top edges up; upright initial still period.',
             'windows_total':len(x),'windows_passing_motion_gate':int(usable.sum()),
             'candidate_activity_counts':dict(Counter(CLASSES[v] for v in labels[usable])),
             'supported_time_range_s': [float(timeline[counts>0].min()),float(timeline[counts>0].max())] if usable.any() else None,
             'raw_convention_angle_ranges_degrees':{name:[float(np.nanmin(avg[:,j])),float(np.nanmax(avg[:,j]))]
                                                    for j,name in enumerate(TARGETS)} if usable.any() else {},
             'input_diagnostics':diagnostics,'acquisition':info,
             'limitations':['No reference joint-angle measurement: cannot calculate phone angle error.',
                            'Walking is the user-supplied trial label; model decisions were not tuned to it.',
                            'Five known locomotion classes only. No validated unknown rejection or standing class.',
                            'Signal-derived orientation frame is not anatomical calibration; browser sensors differ from lab IMUs.',
                            'Overlapping windows are correlated. Model scores and overlap ranges are not confidence intervals.',
                            'Hip and ankle inferred from thigh/shank patterns; no pelvis or foot measurement.']}
    write_json(output/'summary.json',summary)
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    for i,c in enumerate(CLASSES):
        axes[0].plot(timeline[starts][usable]+(LENGTH-1)/HZ/2,p['classifier'][usable,i],'.-',label=c.replace('_',' '))
    axes[0].set(ylabel='Uncalibrated model score',title='Exploratory Camargo two-sensor predictions on your phones',ylim=(0,1.02))
    axes[0].legend(ncol=3,fontsize=8)
    for j,label in enumerate(['Hip (raw sign)','Knee flexion (-raw knee)','Ankle (raw sign)']):
        sign=-1 if j==1 else 1
        axes[1].plot(timeline,avg[:,j]*sign,label=label)
    axes[1].set(xlabel='Session time (s)',ylabel='Predicted angle (degrees)',
                title='No reference angles: these curves have not been validated on phones')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/'phone_predictions.png',dpi=160)
    plt.close(fig)
    lines=['# Exploratory two-sensor Camargo predictions on collected phones','',
           '**These are model outputs, not measured or validated joint angles. No phone angle error can be calculated without a reference.**','',
           f"Motion windows passing the fixed acquisition gate: {int(usable.sum())}/{len(x)}. Candidate activity counts: {summary['candidate_activity_counts']}.",'',
           'Models were trained only on Camargo thigh/shank channels. Phone labels were not used to fit, select epochs, calibrate scores, or choose transforms. Both streams were aligned using recorded common timestamps, interpolated to 60 Hz, low-pass filtered at 20 Hz, and transformed using the same label-free window representation as training. Gyroscope values were converted from deg/s to rad/s; acceleration was normalized by its per-window median magnitude.','',
           'The signal-derived coordinate frame uses mean acceleration and a principal rotation axis. It is rotation-invariant, but it removes absolute segment orientation and is not anatomical calibration. Correct mounting and plausible waveforms do not establish accuracy.','',
           'The first and last still periods are outside the five-class locomotion classifier scope; the fixed motion/axis gate omits unsupported windows. The gate is not a learned standing class or a validated out-of-distribution detector. Saved NPZ files retain all raw model outputs for audit, including rejected windows; use the usable mask.','',
           'Angle CSV uses original Camargo/OpenSim signs. The plot reverses the knee sign so knee flexion is positive. Hip and ankle estimates rely on learned correlations because no pelvis/foot sensor is present. Predictions from overlapping windows are averaged; overlap disagreement is not a confidence interval.','',
           '[Prediction plot](phone_predictions.png) | [Activity windows](activity_windows.csv) | [Experimental angles](joint_angles_experimental.csv) | [Full metadata](summary.json)','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_diagnostics','acquisition']},indent=2))
    print('Saved',output)


if __name__=='__main__':
    main()
