import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import pickle
import json
# Load All Files

with open('final_pipeline.pkl', 'rb') as file_1:
  final_pipeline = pickle.load(file_1)

with open('Drop_Columns.txt', 'r') as file_2:
  Drop_Columns = json.load(file_2)

model_seq2 = load_model('model_seq2.h5')

def run():
  with st.form(key='Customer_Churn_Prediction'):
      st.title('Customer Satisfaction Survey')
      user_id = st.text_input('ID',value='972706cb0db0068e')
      age = st.number_input('Age',min_value=0,max_value=99,value=46)
      gender = st.radio('Gender',('Male','Female'))
      if gender=='Male':
            gender='M'
      else: gender='F'
      region_category = st.radio('Region Category',('Town', 'City','Village'))
      membership_category = st.radio('Membership Category',('Premium Membership','Basic Membership','No Membership', 'Gold Membership','Silver Membership','Platinum Membership'))
      joining_date = st.date_input('Joining Date',datetime.date(2015,3,27),help='YYYY-MM-DD')
      joined_through_referral = st.radio('Did you join using the referral code?',('No','Yes'))
      preferred_offer_types = st.radio('What is your preferred offer types?',('Credit/Debit Card Offers','Gift Vouchers/Coupons','Without Offers'))
      medium_of_operation = st.radio('What device do you usually use?',('Smartphone','Desktop','Both'))
      internet_option = st.radio('What type of network connection do you usually use?',('Mobile_Data','Wi-Fi','Fiber_Optic'))
      last_visit_time = st.text_input('When was the last time you visited?',value='09:41:40',help='HH:mm:ss')
      days_since_last_login = st.number_input('Days Since Last Login',min_value=0,max_value=31,value=16)
      avg_time_spent = st.number_input('Average Time Spent on the Website',step=0.01,format="%.2f",min_value=0.00,max_value=9999.99,value=1447.39)
      avg_transaction_value = st.number_input('Average Transaction Value',step=0.01,format="%.2f",min_value=0.00,max_value=99999.99,value=11839.58)
      avg_frequency_login_days = st.number_input('Frequency of logins per day',min_value=1, max_value=99,value=29)
      points_in_wallet = st.number_input('Points Balance',step=0.01,format="%.2f",min_value=0.00,max_value=9999.99,value=727.91)
      used_special_discount = st.radio('Have you ever used a special discount offer?',('No','Yes'))
      offer_application_preference = st.radio('Do you prefer offers through an application?',('No','Yes'))
      past_complaint = st.radio('Have you ever raised any complaints before ?',('No','Yes'))
      complaint_status = st.radio('Was the complaint resolved ?',('Not Applicable ','Unsolved','Solved','Solved in Follow-up','No Information Available'),help='Select "Not Applicable" if you have never raised a complaint.')
      feedback = st.radio('Any feedback for us?',('No reason specified','Poor Product Quality','Too many ads', 'Poor Website', 'Poor Customer Service', 'Reasonable Price', 'User Friendly Website', 'Products always in Stock', 'Quality Customer Care'))

      submitted = st.form_submit_button('Is the customer at risk of churning ? :thinking_face:')

  df_inf = {
      'user_id': user_id,
      'age': age,
      'gender': gender,
      'region_category': region_category,
      'membership_category': membership_category,
      'joining_date': joining_date,
      'joined_through_referral': joined_through_referral,
      'preferred_offer_types': preferred_offer_types,
      'medium_of_operation': medium_of_operation,
      'internet_option': internet_option,
      'last_visit_time':last_visit_time,
      'days_since_last_login':days_since_last_login,
      'avg_time_spent':avg_time_spent,
      'avg_transaction_value':avg_transaction_value,
      'avg_frequency_login_days':avg_frequency_login_days,
      'points_in_wallet':points_in_wallet,
      'used_special_discount':used_special_discount,
      'offer_application_preference':offer_application_preference,
      'past_complaint':past_complaint,
      'complaint_status':complaint_status,
      'feedback':feedback

  }

  df_inf = pd.DataFrame([df_inf])
  # Data Inference
  df_inf_copy = df_inf.copy()
  

  # Removing unnecessary features
  df_inf_final = df_inf_copy.drop(Drop_Columns,axis=1).sort_index()
  data_inf_transform = final_pipeline.transform(df_inf_final)
  
  st.dataframe(df_inf_final)

  if submitted:
      # Predict using Neural Network
      y_pred_inf = model_seq2.predict(data_inf_transform)
      #st.write('# Is the customer at risk of churning ? :thinking_face:')
      if y_pred_inf == 0:
         st.subheader('Yes, the customer is at risk of churning :disappointed: ')
      else:
         st.subheader('No, the customer is not at risk of churning :wink:')

if __name__ == '__main__':
    run()