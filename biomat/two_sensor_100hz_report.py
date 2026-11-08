"""Recompute saved Camargo metrics and report the two-sensor experiment."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .two_sensor_100hz import OUT, ROOT, reg_score
from .classifier import CLASSES, score


def main():
    result=json.loads((OUT/'results.json').read_text())
    for task in ['classifier','regressor']:
        with np.load(OUT/'final'/task/'predictions.npz',allow_pickle=False) as a:
            if task=='classifier':
                np.testing.assert_allclose(a['predictions'].sum(1),1,atol=1e-6)
                actual=score(a['labels'],a['predictions'],a['subjects'])
            else:
                actual=reg_score(a['angles'],a['predictions'],a['subjects'],a['activity'])
            if actual != result['test'][task]:
                raise ValueError('Saved final metrics do not match predictions')
    classifier,regressor=result['test']['classifier'],result['test']['regressor']
    cv_accuracy=np.mean([r['classifier']['accuracy'] for r in result['cv']])
    cv_rmse=np.mean([r['regressor']['rmse_degrees'] for r in result['cv']],axis=0)
    lines=['# Camargo thigh/shank models at 100 Hz','',
           'Both models are trained from scratch using only right shank and right thigh IMU data. '
           'Foot data, phone recordings and phone labels do not enter training or model selection.','',
           f"Final Camargo test: **{classifier['accuracy']:.2%} classification accuracy**, "
           f"macro-F1 **{classifier['macro_f1']:.4f}**; {classifier['windows']} full windows from AB23/AB24/AB25.",'',
           '| Joint | Test RMSE (degrees) | Test MAE (degrees) | Mean CV RMSE (degrees) |',
           '| --- | ---: | ---: | ---: |']
    for j,name in enumerate(['Hip','Knee','Ankle']):
        lines.append(f"| {name} | {regressor['rmse_degrees'][j]:.2f} | {regressor['mae_degrees'][j]:.2f} | {cv_rmse[j]:.2f} |")
    lines+=['',f"Five-fold mean classification accuracy: {cv_accuracy:.2%}. "
            f"Final BioMAT training duration: {result['final_regressor_epochs']} epochs, chosen before final testing.",'',
            '| Activity | Precision | Recall | F1 | Windows |','| --- | ---: | ---: | ---: | ---: |']
    for c in CLASSES:
        r=classifier['classification_report'][c]
        lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines+=['','| Subject | Classifier accuracy | Hip RMSE | Knee RMSE | Ankle RMSE |',
            '| --- | ---: | ---: | ---: | ---: |']
    for s,r in classifier['subjects'].items():
        v=regressor['by_subject_rmse_degrees'][s]
        lines.append(f"| {s} | {r['accuracy']:.2%} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} |")
    lines+=['','## Protocol','',
            'Input order: shank then thigh, each with acceleration XYZ and angular velocity XYZ. '
            'Full Camargo windows only; first 250 samples at nominal 100 Hz are filtered at 20 Hz and retained '
            'at 100 Hz (2.5 seconds), with no conversion to 60 Hz. Angle targets retain their cached 100 Hz values. '
            'The earlier window preparation and ground-truth-based segment exclusions are inherited. '
            'This is a separate two-sensor experiment, not an exact BioMAT paper reproduction.','',
            'Both sources use the same per-window signal-derived coordinate frame: normalize acceleration '
            'by its median vector magnitude; use mean acceleration for one axis and transverse gyroscope PCA '
            'for a perpendicular axis; choose its sign from the largest absolute projection. '
            'This removes fixed mounting rotation and acceleration units, but also removes absolute segment '
            'orientation. It is not an anatomical calibration. The original plotting script labels acceleration '
            'as m/s^2 despite the stored magnitudes being around one at rest; this experiment avoids relying on that label. '
            'Gyroscope units are rad/s according to scripts.zip/lib/yLabelSelector.m.','',
            'Classification: MovementCNN with 12 inputs, 5 classes, three convolution blocks, 30 fixed epochs, '
            'AdamW lr 0.001 and weight decay 0.0001, batch 128, class-weighted cross entropy. '
            'Regression: BioMAT with 12 inputs, sequence length 250, 4 encoder layers, embedding 256, 16 heads, '
            'feedforward 128, dropout 0.5, three original angle heads; Adam lr 0.001, weight decay 0.001, batch 50.','',
            'Five subject folds within AB06-AB21, training-only standardization. Regression uses up to 50 epochs '
            'and patience 8; each fold retains the best subject-mean joint RMSE checkpoint. Final regression trains '
            'for the ceiling of the median selected epoch on all 16 development participants. '
            'AB23/24/25 are evaluated after selection, but were used in earlier experiments; these are not a fresh study cohort. '
            'The three extra subjects remain unused.','',
            '## Interpretation','',
            'These errors measure the Camargo lab data, not phone accuracy. Hip and ankle estimates can exploit '
            'gait correlations without measuring the pelvis or foot, but may fail on unusual motion. '
            'The classifier covers curated walking/ramp/stair windows only, not standing, running, or unknown activities. '
            'Windows overlap and are not independent observations. Canonicalization is noncausal and uses a complete window. '
            'No confidence interval or phone joint-angle error is available.','',
            'All final metrics were independently recomputed from saved predictions. '
            'Checkpoints and protocol: `Camargo et al. Dataset/two_sensor/canonical100_v1/`. '
            'Final models: `final/classifier/model.pt` and `final/regressor/model.pt`.','',
            '[Test confusion matrix and angle errors](two_sensor_100hz_results.png)','']
    (ROOT/'reports/two_sensor_100hz_results.md').write_text('\n'.join(lines),encoding='utf-8')
    cm=np.asarray(classifier['confusion_matrix'])
    fig,(ax,bars)=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    ax.imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
    labels=['Walk','Ramp up','Ramp down','Stairs up','Stairs down']
    ax.set_xticks(range(5),labels,rotation=35); ax.set_yticks(range(5),labels)
    ax.set(xlabel='Predicted activity',ylabel='Actual activity',title=f"Two sensors: {classifier['accuracy']:.1%} test accuracy")
    for i in range(5):
        for j in range(5):
            value=cm[i,j]/cm[i].sum()
            ax.text(j,i,f'{cm[i,j]}\n{value:.0%}',ha='center',va='center',fontsize=8,color='white' if value>.55 else 'black')
    values=regressor['rmse_degrees']
    bars.bar(['Hip','Knee','Ankle'],values,color=['#3575b9','#db8a20','#419770'])
    for i,v in enumerate(values):
        bars.text(i,v+.15,f'{v:.2f}',ha='center')
    bars.set(ylabel='RMSE (degrees)',title='Camargo test errors; not phone errors',ylim=(0,max(values)*1.25))
    fig.savefig(ROOT/'reports/two_sensor_100hz_results.png',dpi=160)
    plt.close(fig)
    print('Saved report; all metrics independently verified.')
    print(json.dumps({'classifier_accuracy':classifier['accuracy'],'rmse_degrees':values,
                      'cv_accuracy':float(cv_accuracy),'cv_rmse':cv_rmse.tolist()},indent=2))


if __name__=='__main__':
    main()
