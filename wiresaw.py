import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from glob import glob
from src.process import to_dict, get_golden_tool
from src.graph import plot_cluster, plot_pie_chart, get_color, plot_shape
#from src.graph import *
from src.misc import mode, get_fname
import os

if 'df_Y' not in st.session_state:
    df_Y = pd.read_csv('./asset/wire_saw_summary/trend/wire_saw_test.csv', low_memory=False)
    df_Y['BASE_DT'] = df_Y['BASE_DT'].map(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').date())
    df_Y['INGOT_LEN'] = round(df_Y['INGOT_LEN'], 1)
    df_Y.loc[df_Y['WAIT_TIME'] == '_', 'WAIT_TIME'] = None
    df_Y['WAIT_TIME'] = df_Y['WAIT_TIME'].astype('float')
    df_Y.loc[(df_Y['WAIT_TIME'] < 0), 'WAIT_TIME'] = 0
    st.session_state['df_Y'] = df_Y
    

if 'df_shape' not in st.session_state:
    st.session_state['df_shape'] = pd.read_csv('./asset/wire_saw_summary/shape/shape_data_2401_2409.csv')
    st.session_state['df_shape'] = st.session_state['df_shape'].merge(st.session_state['df_Y'][['LOT_ID','EQP_NM', 'BASE_DT']].dropna().drop_duplicates(), on='LOT_ID', how='left') 

if 'df_cluster' not in st.session_state:
    st.session_state['df_cluster'] = pd.read_csv('./asset/wire_saw_summary/cluster/cluster.csv')

if 'list_recipe' not in st.session_state:
    st.session_state['list_recipe'] = st.session_state['df_Y'].RECIPE_ID.unique().tolist()

if 'list_ingot_length_range' not in st.session_state:
    st.session_state['list_ingot_length_range'] = np.round(list(np.arange(st.session_state['df_Y'].INGOT_LEN.min(), st.session_state['df_Y'].INGOT_LEN.max(), 0.1)), 1)
if 'list_own_time_range' not in st.session_state:
    st.session_state['list_down_time_range'] = list(np.arange(st.session_state['df_Y'].WAIT_TIME.min(), 601))

if 'dict_cluster' not in st.session_state:
    st.session_state['dict_cluster'] = to_dict(st.session_state['df_cluster'])

st.set_page_config(
    page_title = 'WIRE SAW - WARP',
    page_icon = '📊',
    initial_sidebar_state = 'collapsed',
    layout = 'wide'
)


start_date = st.sidebar.date_input('Start date', datetime.date.today() - datetime.timedelta(days=60))
end_date = st.sidebar.date_input('End date', datetime.date.today())

Ingot_length_min, Ingot_length_max = st.sidebar.select_slider('Ingot_length',
                                        options=st.session_state['list_ingot_length_range'],
                                        value=(35.0,40.0)
                                        )
Down_time_min, Down_time_max = st.sidebar.select_slider('WAIT_TIME',  
                                    options=st.session_state['list_down_time_range'],
                                    value=(0,80)
                                    )
list_recipe = st.sidebar.multiselect(
    label = 'Recipe',
    options = st.session_state['list_recipe'],
    default   = ['10-133', '11-133']
    )


list_lot_id = st.session_state['df_Y'][(st.session_state['df_Y'].INGOT_LEN.between(Ingot_length_min, Ingot_length_max, inclusive='both'))  
                                    & (st.session_state['df_Y'].WAIT_TIME.between(Down_time_min, Down_time_max, inclusive='both')) 
                                    & (st.session_state['df_Y'].BASE_DT.between(start_date, end_date, inclusive='both'))
                                    & (st.session_state['df_Y'].RECIPE_ID.isin(list_recipe))
                                    ].LOT_ID.tolist()

st.markdown('## WIRE SAW (WARP)')

# 1. warp trend
st.markdown('---')
st.markdown('### 1. WARP 품질 현황')



df_warp = st.session_state['df_Y'][st.session_state['df_Y'].LOT_ID.isin(list_lot_id)].groupby('EQP_NM').WARP_BF.mean().reset_index().sort_values('WARP_BF')


chart = alt.Chart(df_warp).mark_bar(color = '#E1002A').encode(
    x = alt.X('EQP_NM', title = None, sort = '-y'),
    y = alt.Y('WARP_BF', title = 'WARP (um)', scale = alt.Scale(domain = [6, 16], clamp = True))
)

st.altair_chart(chart, use_container_width = True)

###############################################################

eqps = df_warp.sort_values('EQP_NM')
warps = eqps['WARP_BF'].values
idx_init = int(np.argmax(warps))

###############################################################


# 2. clustering
st.markdown('---')
st.markdown('### 2. Golden Tool 탐색')

col1, col2 = st.columns(2)

with col1:
    eqp_1 = st.selectbox(
        label = '대상 장비',
        options = eqps,
        index = idx_init
    )

    df_score = pd.read_csv(f'./asset/wire_saw_summary/score/{eqp_1}.csv')

with col2:
    eqp_2 = st.selectbox(
        label = 'Golden tool',
        options = get_golden_tool(st.session_state['dict_cluster'], eqp_1),
        index = 0
    )

col3, col4 = st.columns([1, 1.45])

with col3:
    fig = plot_cluster(
        df_cluster = st.session_state['df_cluster'], 
        eqp_1 = eqp_1, 
        eqp_2 = eqp_2
    )
    st.pyplot(fig, use_container_width = True)

with col4:
    st.markdown(f'#### EQP: {eqp_1}')
    fig = plot_shape(st.session_state['df_shape'], eqp_1, list_lot_id )
    st.pyplot(fig, use_container_width = True)

    st.markdown(f'#### EQP: {eqp_2}')
    fig = plot_shape(st.session_state['df_shape'], eqp_2, list_lot_id )
    st.pyplot(fig, use_container_width = True)


# 3. X 중요도
st.markdown('---')
st.markdown('### 3. X 인자 중요도')

col5, _, col6 = st.columns([0.55, 0.05, 1])

with col5:
    score = df_score[pd.to_datetime(df_score.DATE, format = '%Y%m%d').dt.date.between(start_date, end_date, inclusive='both')]
    
    df_types = score[['TYPE', 'SCORE']]
    df_types = df_types.groupby(by = 'TYPE').sum()
 
    t_x = df_types.index.tolist()
    t_y = df_types['SCORE'].tolist()

    fig = plot_pie_chart(
        x = t_x,
        y = t_y,
        title = f'EQP: {eqp_1}'
    )

    st.pyplot(fig, use_container_width = True)

with col6:
    types = ['FDC']
    df_fdc = score.query(f'TYPE == {types}')
    df_fdc = df_fdc.groupby(['FEATURE'], as_index=False)['SCORE'].mean()

    df_fdc = df_fdc.sort_values(by = 'SCORE', ascending = False)

    st.markdown('#### FDC 인자')

    chart = alt.Chart(df_fdc.iloc[:10]).mark_bar(color = '#E1002A').encode(
        x = alt.X('SCORE', title = 'IMPORTANCE (%)'),
        y = alt.Y('FEATURE', title = None, sort = '-x', axis = alt.Axis(labelLimit = 200))
    )

    st.altair_chart(chart, use_container_width = True)

    types = ['CoA', 'ETC']
    df_coa = score.query(f'TYPE == {types}')
    df_coa = df_coa.groupby(['FEATURE'], as_index=False)['SCORE'].mean()
    df_coa = df_coa.sort_values(by = 'SCORE', ascending = False)

    st.markdown('#### 원부자재(CoA) / ETC')

    chart = alt.Chart(df_coa.iloc[:10]).mark_bar(color = '#E1002A').encode(
        x = alt.X('SCORE', title = 'IMPORTANCE (%)'),
        y = alt.Y('FEATURE', title = None, sort = '-x', axis = alt.Axis(labelLimit = 200)),
    )

    st.altair_chart(chart, use_container_width = True)


# 4. X 인자 비교
st.markdown('---')
st.markdown('### 4. X 인자 비교 (Golden tool vs 대상 장비)')

# parameters
df_all = df_score.sort_values(by = 'SCORE', ascending = False)
options = df_all['FEATURE'].unique().tolist()
options = list(filter(lambda x: x != 'EQP', options))

param = st.selectbox(
    label = 'Parameter',
    options = options,
    index = 0
)

fig = plt.figure(figsize = (10, 3))

# reference
r_ts = []
r_vs = []

no_key_error_flag = False
for path in glob(f'./asset/wire_saw_summary/fdc/{eqp_1}/*.csv'):
    path = path.replace('\\','/')
    if os.path.split(os.path.splitext(path)[0])[1] in list_lot_id:
        df = pd.read_csv(path)
        if param in df:
            df['BASE_DT'] = pd.to_datetime(df['BASE_DT'].astype(str))
            r_ts.append(mode(df['BASE_DT']))
            r_vs.append(df[param])
            

if len(r_ts):
    r_ts = np.array(r_ts)
    r_ts = r_ts - np.min(r_ts)
    r_ts = r_ts / np.timedelta64(1, 's')
    r_ts = r_ts / (np.max(r_ts) + 1e-8)

    index = np.argsort(r_ts)
    r_ts = [r_ts[i] for i in index]
    r_vs = [r_vs[i] for i in index]
    for t, v in zip(r_ts, r_vs):
        color = get_color(t, 'Reds') if len(r_ts) > 1 else 'r'
        if t == np.max(r_ts):
            plt.plot(v, color = color, lw = 2, label = eqp_1)
        else:
            plt.plot(v, color = color)
else:
    st.write(f':red[REFERENCE does not have "{param}" info.]')

# golden tool
g_ts = []
g_vs = []
for path in glob(f'./asset/wire_saw_summary/fdc/{eqp_2}/*.csv'):
    path = path.replace('\\','/')
    if os.path.split(os.path.splitext(path)[0])[1] in list_lot_id:
        df = pd.read_csv(path)
        if param in df:
            df['BASE_DT'] = pd.to_datetime(df['BASE_DT'].astype(str))
            g_ts.append(mode(df['BASE_DT']))
            g_vs.append(df[param])    

if len(g_ts):
    g_ts = np.array(g_ts)
    g_ts = g_ts - np.min(g_ts)
    g_ts = g_ts / np.timedelta64(1, 's')
    g_ts = g_ts / (np.max(g_ts) + 1e-8)

    index = np.argsort(g_ts)
    g_ts = [g_ts[i] for i in index]
    g_vs = [g_vs[i] for i in index]
    for t, v in zip(g_ts, g_vs):
        color = get_color(t, 'Blues') if len(g_ts) > 1 else 'b'        
        if t == np.max(g_ts):
            plt.plot(v, color = color, lw = 2, label = eqp_2)
        else:
            plt.plot(v, color = color)
else:
    st.write(f':red[GOLDEN TOOL does not have "{param}" info.]')

plt.xlim(left = 0)
plt.xlabel('RUNTIME (min)')
plt.ylabel(param)
plt.legend()

if len(r_ts) or len(g_ts):
    st.pyplot(fig, use_container_width = True)
