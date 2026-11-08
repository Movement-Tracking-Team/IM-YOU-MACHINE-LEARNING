"""Run with: python -m streamlit run dashboard/app.py"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.backend import REPORT, CHECKPOINTS, evaluate, load_report, summarize, window_table
from biomat.classifier import CLASSES

st.set_page_config(page_title='Motion · Model validation', page_icon='🚶', layout='wide')
st.markdown('''<style>
.block-container {max-width:1280px;padding-top:2.2rem;}
[data-testid="stMetric"] {background:#13283c;border:1px solid #29435a;border-radius:12px;padding:16px;}
.eyebrow {color:#64d8c4;letter-spacing:.16em;font-size:.78rem;font-weight:700;}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">MOTION LAB / CAMARGO</div>', unsafe_allow_html=True)
st.title('Model validation')
st.caption('100 Hz · 250 samples per window (2.5 s) · 12 channels · right shank + thigh')

with st.sidebar:
    st.header('Evaluation')
    selected_model = st.radio('Activity classifier', ['adapted', 'original'],
                             format_func=lambda x: {'adapted': 'Phone-adapted', 'original': 'Original Camargo'}[x])
    st.caption('Joint angles always use the unchanged 100 Hz BioMAT model.')
    device = st.selectbox('Inference device', ['auto', 'cpu', 'cuda'])
    run = st.button('Run saved models again', type='primary', use_container_width=True)
    st.caption('Runs inference on AB23, AB24 and AB25. Does not train the models.')

if run:
    status = st.empty()
    try:
        with st.spinner('Evaluating held-out Camargo data…'):
            evaluate(device, progress=status.info)
        status.success('Fresh validation saved.')
    except Exception as error:
        st.error(f'Validation could not finish: {error}')

if not (REPORT / 'summary.json').exists():
    st.info('Use “Run saved models again” to create the first validation report.')
    st.stop()

@st.cache_data(show_spinner=False)
def read_results(report_version):
    return load_report()

try:
    data, summary = read_results((REPORT / 'summary.json').stat().st_mtime_ns)
except Exception as error:
    st.error(f'Cannot read validation report: {error}')
    st.stop()

with st.sidebar:
    subjects = st.multiselect('Held-out subjects', sorted(set(data['subjects'])), default=sorted(set(data['subjects'])))
    activities = st.multiselect('Reference activity', CLASSES, default=CLASSES,
                               format_func=lambda x: x.replace('_', ' ').title())
    st.divider()
    st.caption('Report generated ' + summary['provenance']['created_utc'])

mask = np.isin(data['subjects'], subjects) & np.isin(data['labels'], [CLASSES.index(v) for v in activities])
if not mask.any():
    st.info('Select at least one subject and an activity with recorded windows.')
    st.stop()
metrics = summarize(data, mask)
table = window_table(data)
filtered = table.loc[mask]
model_metrics = metrics[selected_model]
other = 'original' if selected_model == 'adapted' else 'adapted'
st.info('These subjects were excluded from training, but their results have been reviewed in earlier experiments. '
        'This is a repeat validation, not a new independent test. Overlapping windows are correlated.')
cols = st.columns(4)
cols[0].metric('Activity accuracy', f"{model_metrics['accuracy']:.2%}",
               f"{100 * (model_metrics['accuracy'] - metrics[other]['accuracy']):+.2f} pp vs {other}")
cols[1].metric('Macro F1 · five classes', f"{model_metrics['macro_f1']:.3f}")
cols[2].metric('Evaluation windows', f'{int(mask.sum()):,}')
cols[3].metric('Held-out subjects', len(set(data['subjects'][mask])))

overview, explorer, details = st.tabs(['Performance', 'Explore predictions', 'Method & downloads'])
with overview:
    st.subheader('Activity recognition')
    comparison = pd.DataFrame([{'Model': name.title(), 'Accuracy (%)': metrics[name]['accuracy'] * 100,
                               'Macro F1': metrics[name]['macro_f1']} for name in ('original', 'adapted')])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption('Macro F1 uses all five classes, including classes excluded by a filter. All metrics reflect the current filters.')
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('**Confusion matrix**')
        st.caption('Rows: reference activity · Columns: predicted activity')
        confusion = pd.DataFrame(model_metrics['confusion_matrix'], index=CLASSES, columns=CLASSES)
        st.dataframe(confusion.style.background_gradient(cmap='Blues').format('{:.0f}'), use_container_width=True)
    with right:
        st.markdown('**Per-class performance**')
        per_class = pd.DataFrame({name: model_metrics['classification_report'][name] for name in CLASSES}).T
        st.dataframe(per_class.style.format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}', 'support': '{:.0f}'}), use_container_width=True)
    st.subheader('Joint-angle error')
    st.caption('BioMAT predictions versus Camargo reference angles. Lower error is better; units are degrees.')
    for col, joint, value in zip(st.columns(3), ['Hip', 'Knee', 'Ankle'], metrics['angles']['rmse_degrees']):
        col.metric(joint + ' RMSE', f'{value:.2f}°')
    subject_errors = pd.DataFrame(metrics['angles']['by_subject_rmse_degrees'], index=['Hip', 'Knee', 'Ankle']).T
    st.dataframe(subject_errors.style.format('{:.2f}°'), use_container_width=True)

with explorer:
    st.subheader('Inspect a prediction window')
    errors_only = st.checkbox('Show activity classification errors only')
    choices = filtered.loc[~filtered[selected_model + '_correct']] if errors_only else filtered
    if choices.empty:
        st.success('No classification errors in this selection.')
    else:
        window_id = st.selectbox('Window', choices['window_id'].tolist(),
                                format_func=lambda i: f"{i} · {table.loc[i, 'subject']} · {table.loc[i, 'reference'].replace('_', ' ')}")
        row = table.loc[window_id]
        st.write(f"Reference: **{row['reference'].replace('_', ' ')}** · Prediction: **{row[selected_model + '_prediction'].replace('_', ' ')}**")
        st.bar_chart(pd.DataFrame({'Model probability': data[selected_model][window_id]}, index=CLASSES))
        st.caption('Softmax probabilities are model scores; they are not calibrated reliability estimates.')
        joint = st.selectbox('Joint angle', ['Hip', 'Knee', 'Ankle'])
        j = ['Hip', 'Knee', 'Ankle'].index(joint)
        curves = pd.DataFrame({'Time within window (s)': np.arange(250) / 100,
                               'Camargo reference (°)': data['truth'][window_id, :, j],
                               'BioMAT prediction (°)': data['angles'][window_id, :, j]})
        st.line_chart(curves, x='Time within window (s)', y=['Camargo reference (°)', 'BioMAT prediction (°)'])
        st.caption(f"Window {joint.lower()} RMSE: {row[joint.lower() + '_rmse_degrees']:.2f}°. Original target sign conventions are preserved.")
        st.download_button('Download this angle curve', curves.to_csv(index=False), f'camargo_window_{window_id}_{joint.lower()}.csv', 'text/csv')
    st.dataframe(filtered, hide_index=True, use_container_width=True)

with details:
    st.subheader('What is being tested')
    st.markdown('''The original activity classifier was trained on Camargo subjects **AB06–AB21**.
The phone-adapted classifier was subsequently fine-tuned using phone recordings and Camargo training data.
Both are evaluated here on **AB23, AB24 and AB25**. The joint-angle model is the same saved 100 Hz BioMAT model for both comparisons.

Each input contains **250 × 12** values: three acceleration axes and three gyroscope axes for the shank, followed by the thigh.
The prepared Camargo cache retains its nominal **100 Hz** grid. Each full 256-sample cached window is cropped to its first 250 samples,
filtered with a fourth-order **20 Hz low-pass** filter and transformed with the training pipeline's signal-derived orientation basis.
Acceleration is divided by its per-window median magnitude, making it dimensionless; gyroscope inputs retain radians per second
before the saved training mean and standard deviation are applied. No scaler is fitted on test data.

This page validates prepared Camargo windows. It does not establish performance on new phone recordings or live feedback.
The orientation basis is signal-derived, not an anatomical calibration. Windows overlap and cannot be treated as independent trials.''')
    st.download_button('Download selected window results', filtered.to_csv(index=False), 'camargo_selected_windows.csv', 'text/csv')
    st.download_button('Download full validation report', json.dumps(summary, indent=2), 'camargo_validation.json', 'application/json')
    st.download_button('Download selected metrics', json.dumps(metrics, indent=2), 'camargo_selected_metrics.json', 'application/json')
    with st.expander('Checkpoint and data provenance'):
        st.json(summary['provenance'])
