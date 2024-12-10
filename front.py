import streamlit as st
base_url = 'http://10.150.9.121/tttm_go_back_address'

st.set_page_config(
    page_title = 'SMART TTTM',
    page_icon = '📊',
    initial_sidebar_state = 'collapsed',
    layout = 'wide'
)

def set_title(title):
    return f"<h2 style = 'text-align: center; color: black;'>{title}</h1>"

st.image('./asset/wire_saw_summary/front/banner.PNG', use_column_width = True)
st.markdown('---')


col11, col12, col13, col14 = st.columns(4)

with col11:
    st.markdown(set_title('WIRE SAW'), unsafe_allow_html = True)

with col12:
    st.markdown(set_title('DSP'), unsafe_allow_html = True)

with col13:
    st.markdown(set_title('FCS'), unsafe_allow_html = True)

with col14:
    st.markdown(set_title('EPI'), unsafe_allow_html = True)


col21, col22, col23, col24 = st.columns(4)

with col21: st.image('./asset/wire_saw_summary/front/wiresaw.gif')
with col22: st.image('./asset/wire_saw_summary/front/dsp.gif')
with col23: st.image('./asset/wire_saw_summary/front/fcs.gif')
with col24: st.image('./asset/wire_saw_summary/front/epi.gif')


col31, col32, col33, col34 = st.columns(4)

with col31:
    is_WARP_clicked = st.button(
        label = 'WARP', 
        use_container_width = True
    )
    if is_WARP_clicked:
        st.switch_page(f'./pages/wiresaw.py')

    is_BOW_clicked = st.button(
        label = 'BOW', 
        use_container_width = True, 
        disabled = True
    )
    if is_BOW_clicked:
        #st.switch_page(f'./pages/wiresaw.py')
        pass

    is_NANO_clicked = st.button(
        label = 'NANO', 
        use_container_width = True, 
        disabled = True
    )
    if is_NANO_clicked:
        #st.switch_page(f'./pages/wiresaw.py')
        pass

with col32:
    st.link_button(
        label = 'GBIR', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )
    st.link_button(
        label = 'SFQR', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )
    st.link_button(
        label = 'ESFQR', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )

with col33:
    st.link_button(
        label = 'LLS (47 nm)', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )
    st.link_button(
        label = 'METAL', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )

with col34:
    st.link_button(
        label = 'SFQR', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )

    st.link_button(
        label = 'SITE NT', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )
    st.link_button(
        label = 'DELTA TEMP', 
        url = '', 
        use_container_width = True, 
        disabled = True
    )


for _ in range(9):
    st.write('')
#st.write('해당 페이지는 품질 현황과 장비의 TTTM Point 제안을 위해 개발되었습니다.')
st.write('문의사항: DS1팀 이주영T, DS1팀 한승철P, DS2팀 유광남P')
