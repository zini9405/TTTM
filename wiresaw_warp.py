import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import json
#from src.misc import load_json
from src.graph import get_color

import json
def load_json(
    path: str
) -> dict:
    
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

if 'summary' not in st.session_state:
    summary = pd.read_csv('./asset/wire_saw_eqp/summary/summary.csv', low_memory=False)
    summary['DATE'] = pd.to_datetime(summary['DATE'].astype(str))
    summary = summary.sort_values(by = 'DATE')
    summary = summary.reset_index(drop = True)
    st.session_state['summary'] = summary
if 'now' not in st.session_state:
    st.session_state['now'] = datetime.now()

if 'df_Y' not in st.session_state:
    st.session_state['df_Y'] = pd.read_csv('./asset/wire_saw_summary/trend/wire_saw_test.csv', low_memory=False)
    # st.session_state['df_Y']['BASE_DT'] = st.session_state['df_Y']['BASE_DT'].map(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    # st.session_state['df_Y']['INGOT_LEN'] = round(st.session_state['df_Y'].INGOT_LEN, 1)
    # st.session_state['df_Y'].loc[st.session_state['df_Y'].WAIT_TIME == '_', 'WAIT_TIME'] = None
    # st.session_state['df_Y']['WAIT_TIME'] = st.session_state['df_Y'].WAIT_TIME.astype('float')

if 'warp_json' not in st.session_state:
    st.session_state['warp_json'] = load_json('./asset/wire_saw_warp/warp.json')


st.set_page_config(layout = 'wide')


col11, col12, col13 = st.columns([1, 1, 2])

with col11:
    st.subheader('LOT 별 Warp 예측값')
    lots = sorted(st.session_state['warp_json'].keys())

    lot = st.selectbox(
        label = 'LOT ID',
        options = lots,
        index = 0
    )


col21, col22 = st.columns(2)
interval_selector = alt.selection_interval('interval_selection')

with col21:
    # warp
    pred = np.array(st.session_state['warp_json'][lot]['PRED'])
    target = np.zeros_like(pred) if st.session_state['warp_json'][lot]['TARGET'] is None else np.array(st.session_state['warp_json'][lot]['TARGET'])
    total = np.stack((pred, target), axis=1)
    chart_data = pd.DataFrame(total, columns=["Prediction", "Target"])
    st.line_chart(chart_data, color=["#04f", "#f00"], x_label='Block Position', y_label='Warp (nm)')


col31, col32 = st.columns(2)

with col31:
    st.markdown('##### LOT 정보')
    st.write(st.session_state['df_Y'].loc[st.session_state['df_Y'].LOT_ID == lot])
