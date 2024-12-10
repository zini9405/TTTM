import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
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
    st.session_state['df_Y'] = pd.read_csv('./asset/wire_saw_summary/trend/240923_wire_saw.csv', low_memory=False)
    st.session_state['df_Y']['BASE_DT'] = st.session_state['df_Y']['BASE_DT'].map(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    st.session_state['df_Y']['INGOT_LEN'] = round(st.session_state['df_Y'].INGOT_LEN, 1)
    st.session_state['df_Y'].loc[st.session_state['df_Y'].WAIT_TIME == '_', 'WAIT_TIME'] = None
    st.session_state['df_Y']['WAIT_TIME'] = st.session_state['df_Y'].WAIT_TIME.astype('float')
st.set_page_config(layout = 'wide')


col11, col12, col13 = st.columns([1, 1, 2])

with col11:
    eqps = sorted(st.session_state['df_Y']['EQP_NM'].unique())

    eqp = st.selectbox(
        label = 'EQP_NM',
        options = eqps,
        index = 0
    )

    df_eqp = st.session_state['df_Y'].loc[st.session_state['df_Y'].EQP_NM == eqp].sort_values('BASE_DT')

with col12:
    _t_i = df_eqp['BASE_DT'].iloc[0]
    _t_f = df_eqp['BASE_DT'].iloc[-1]

    d = st.date_input(
        label = 'TIME',
        value = (_t_i, _t_f),
        min_value = _t_i,
        max_value = _t_f,
        format = 'YYYY/MM/DD',
    )

    t_i, t_f = d
    #t_i = pd.to_datetime(t_i)
    #t_f = pd.to_datetime(t_f)

    df_eqp = df_eqp.loc[df_eqp['BASE_DT'] >= t_i]
    df_eqp = df_eqp.loc[df_eqp['BASE_DT'] <= t_f]
    df_eqp = df_eqp.reset_index(drop = True)


col21, col22 = st.columns(2)

interval_selector = alt.selection_interval('interval_selection')

with col21:
    df_eqp_grp = df_eqp.groupby(['BASE_DT', 'LOT_ID']).agg({'WARP_BF':'mean', 'BOW_BF':'mean', 'TTV':'mean'}).reset_index()
    # warp
    chart = alt.Chart(df_eqp_grp).mark_bar(color = '#E1002A').encode(
        x = alt.X('LOT_ID', title = None, sort = alt.EncodingSortField('BASE_DT', order = 'ascending')),
        y = alt.Y('WARP_BF', title = 'WARP (um)'),
        tooltip = ['BASE_DT', 'LOT_ID', 'WARP_BF', 'BOW_BF', 'TTV']
    ).add_params(interval_selector)

    event = st.altair_chart(chart, use_container_width = True, on_select = 'rerun')

    # bow
    chart = alt.Chart(df_eqp_grp).mark_bar(color = '#E1002A').encode(
        x = alt.X('LOT_ID', title = None, sort = alt.EncodingSortField('BASE_DT', order = 'ascending')),
        y = alt.Y('BOW_BF', title = 'BOW (um)'),
        tooltip = ['BASE_DT', 'LOT_ID', 'WARP_BF', 'BOW_BF', 'TTV']
    )

    st.altair_chart(chart, use_container_width = True)

    # ttv
    chart = alt.Chart(df_eqp_grp).mark_bar(color = '#E1002A').encode(
        x = alt.X('LOT_ID', title = None, sort = alt.EncodingSortField('BASE_DT', order = 'ascending')),
        y = alt.Y('TTV', title = 'TTV (um)'),
        tooltip = ['BASE_DT', 'LOT_ID', 'WARP_BF', 'BOW_BF', 'TTV']
    )

    st.altair_chart(chart, use_container_width = True)

with col22:
    if 'LOT_ID' in event['selection']['interval_selection']:
        lots = event['selection']['interval_selection']['LOT_ID']
    else:
        lots = []
    #st.write(lots)
    
    if len(lots) > 0:
        datas = [load_json(f'./asset/wire_saw_eqp/y/{lot}.json') for lot in lots]

        # warp
        fig = plt.figure(figsize = (10, 3.8))
        for i, data in enumerate(datas):
            position = np.array(data['SEQ_NO'])
            position = 100 * position / np.max(position)

            index = (i + 1) / len(lots)
            color = get_color(i / len(lots), 'Grays')
            if index == 1:
                plt.plot(position, data['WARP_BF'], color = color, lw = 3)
            else:
                plt.plot(position, data['WARP_BF'], color = color)

        plt.xlim(0, 100)
        plt.xlabel('POSITION (%)')
        plt.ylabel('WARP (um)')
        st.pyplot(fig, use_container_width = True)

        # bow
        fig = plt.figure(figsize = (10, 3.8))
        for i, data in enumerate(datas):
            position = np.array(data['SEQ_NO'])
            position = 100 * position / np.max(position)

            index = (i + 1) / len(lots)
            color = get_color(i / len(lots), 'Grays')
            if index == 1:
                plt.plot(position, data['BOW_BF'], color = color, lw = 3)
            else:
                plt.plot(position, data['BOW_BF'], color = color)

        plt.xlim(0, 100)
        plt.xlabel('POSITION (%)')
        plt.ylabel('BOW (um)')
        st.pyplot(fig, use_container_width = True)

        # ttv
        fig = plt.figure(figsize = (10, 3.8))
        for i, data in enumerate(datas):
            position = np.array(data['SEQ_NO'])
            position = 100 * position / np.max(position)

            index = (i + 1) / len(lots)
            color = get_color(i / len(lots), 'Grays')
            if index == 1:
                plt.plot(position, data['TTV'], color = color, lw = 3)
            else:
                plt.plot(position, data['TTV'], color = color)

        plt.xlim(0, 100)
        plt.xlabel('POSITION (%)')
        plt.ylabel('TTV (um)')
        st.pyplot(fig, use_container_width = True)


# with col13:
#     col131, col132 = st.columns(2)
#     with col131:
#         lots = sorted(df_eqp.LOT_ID.unique())

#         lot = st.selectbox(
#             label = 'LOT_ID',
#             options = lots,
#             index = 0
#         )
#         df_eqp_lot = df_eqp[df_eqp.LOT_ID == lot]

#         chart = alt.Chart(df_eqp_lot).mark_bar(color = '#E1002A').encode(
#             x = alt.X('BLOCK_POS', title = None, sort = alt.EncodingSortField('DATE', order = 'ascending')),
#             y = alt.Y('WARP_BF', title = 'WARP (um)'),
#             tooltip = ['DATE', 'LOT_ID', 'WARP_BF', 'BOW_BF', 'TTV']
#         ).add_params(interval_selector)

#         event = st.altair_chart(chart, use_container_width = True, on_select = 'rerun')

        
