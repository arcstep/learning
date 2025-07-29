import streamlit as st
import numpy as np
import pandas as pd

# 下拉框
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
    df['first column'])

'You selected: ', option

# 复选框
if st.checkbox('Show dataframe'):
    # 数据框
    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))

    st.dataframe(dataframe.style.highlight_max(axis=0))

# 折线图
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# 地图
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(map_data)

# 滑块
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

