import streamlit as st
import eda # python file
import prediction # python file

navigation = st.sidebar.selectbox('Page Navigation: ',('EDA','Customer Churn Prediction'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()

