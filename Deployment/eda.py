import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.preprocessing import LabelEncoder

from PIL import Image

st.set_page_config(
    page_title='Customer Churn Prediction Using Artificial Neural Network in E-commerce Company',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

def run():
    # title
    st.title('Customer Churn Prediction Using Artificial Neural Network in E-commerce Company')
    st.write('by Ahmad Luay Adnani')

    # sub header
    st.subheader ('Exploratory Data Analysis of the Dataset.')

    # Add Image
    image = Image.open('churn.jpeg')
    st.image(image,caption = 'Customer churn illustration')

    # Description
    st.write('In customer relationship management, it is important for e-commerce businesses to attract new customers and retain existing ones. Predicting customer churn in e-commerce business is critical to the success of online retailers. By analyzing customer data, businesses can gain insights into customer behavior and develop strategies to retain customers, ultimately improving customer satisfaction and driving revenue growth.')
    st.write('# Dataset') 
    st.write('Dataset used in this analysis is churn dataset from an e-commerce company that wants to minimize the risk of a customer stopping using the product they offer.')

    # show dataframe
    df = pd.read_csv('churn.csv')
    st.dataframe(df)
    # add description of Dataset
    st.write('Following are the variables and definitions of each column in the dataset.')
    st.write("`user_id` : ID of a customer")
    st.write("`age` : Age of a customer")
    st.write("`gender` : Gender of a customer")
    st.write("`region_category` : Region that a customer belongs to")
    st.write("`membership_category` : Category of the membership that a customer is using")
    st.write("`joining_date` : Date when a customer became a member")
    st.write("`joined_through referral` : Whether a customer joined using any referral code or ID")
    st.write("`preferred_offer types` : Type of offer that a customer prefers")
    st.write("`medium_of operation` : 	Medium of operation that a customer uses for transactions")
    st.write("`internet_option` : Type of internet service a customer uses")
    st.write("`last_visit_time` : The last time a customer visited the website")
    st.write("`days_since_last_login` : Number of days since a customer last logged into the website")
    st.write("`avg_time_spent` : Average time spent by a customer on the website")
    st.write("`avg_transaction_value` : Average transaction value of a customer")
    st.write("`avg_frequency_login_days` : Number of times a customer has logged in to the website")
    st.write("`points_in_wallet` : Points awarded to a customer on each transaction")
    st.write("`used_special_discount` : Whether a customer uses special discounts offered")
    st.write("`offer_application_preference` : Whether a customer prefers offers")
    st.write("`past_complaint` : Whether a customer has raised any complaints")
    st.write("`complaint_status` : Whether the complaints raised by a customer was resolved")
    st.write("`feedback` : Feedback provided by a customer")
    st.write("`churn_risk_score` : Churn score `0` : Not churn `1` : Churn")

    ###
    # Churn Prediction

    st.write('# Exploratory Data Analysis ')
    st.write('## Number of Customer at Risk of Churning')
    
    # churn 
    df_eda = df.copy()
    df_eda.churn_risk_score.replace({0:'Not Churn',1:'Churn'}, inplace=True)
    churn = df_eda.churn_risk_score.value_counts().to_frame().reset_index()
    
    # Plot PieChart with Plotly
    fig = px.pie(churn,values='churn_risk_score', names='index',color_discrete_sequence=['red','blue'])
    fig.update_layout(title_text = "Number of Customer at risk of Churning")
    st.plotly_chart(fig)
    st.write('Based on visualization above, the percentage of customer at risk of churning is 54.1%. Further data exploration is needed to find out what factors cause these customers to be at risk of churning.')

    ###
    # Number of Customers Based on Their Membership Categories
    st.write('## Number of Customers Based on Their Membership Categories')
    
    # membership category
    membership_category = df_eda.groupby(['churn_risk_score','membership_category']).aggregate(Number_of_customer_per_membership_category=('membership_category','count')).reset_index()
    
    # plotting bar plot
    fig = px.bar(membership_category, x="membership_category", y="Number_of_customer_per_membership_category",color='churn_risk_score',color_discrete_sequence=['red','blue'],
             orientation="v",hover_name="membership_category"        
                
             )
    fig.update_layout(title_text = "Number of customers based on their membership category")
    st.plotly_chart(fig)
    st.write('Based on visualization above, customers **without membership** and customers with **basic membership** have the highest risk of churning. Based on my assumption, customers without membership and customers with basic membership may have a higher risk of churning for several reasons:')
    st.write('1. **Lack of loyalty**: Customers without membership or with basic membership may not feel a strong sense of loyalty to the company or brand, making it easier for them to switch to a competitor.')
    st.write('2. **Limited benefits**: Basic membership may offer limited benefits or perks compared to higher-tier memberships, making it less attractive to customers who may be seeking more value.')
    st.write('3. **Price sensitivity**: Customers without membership or with basic membership may be more price-sensitive and may be more likely to switch to a competitor if they find a better deal elsewhere.')
    st.write('4. **Limited engagement**: Customers without membership or with basic membership may have limited engagement with the company or brand, making it harder for the company to build a strong relationship with them and retain their loyalty.')

    ###
    # Average Transaction Value

    st.write('## Average Transaction Value')

    # average transaction value
    avg_transaction_value = df_eda.groupby(['churn_risk_score']).aggregate(avg_transaction_value=('avg_transaction_value','mean')).reset_index()
    
    # plotting bar plot
    fig = px.bar(avg_transaction_value, x="churn_risk_score", y="avg_transaction_value",color='churn_risk_score',color_discrete_sequence=['red','blue'],
             orientation="v"       
                
             )
    fig.update_layout(title_text = "Average Transaction Value")
    st.plotly_chart(fig)
    st.write('Based on visualization above, customers who are at risk of churning have a **lower average transaction value** compared to customers who are not at risk of churning. Based on my assumption, customers who are at risk of churning may have a lower average transaction value for several reasons:')
    st.write("1. **Reduced usage**: Customers who are at risk of churning may be using the company's products or services less frequently or may have stopped using them altogether. This reduced usage can result in a lower average transaction value.")
    st.write('2. **Price sensitivity**: Customers who are at risk of churning may be more price-sensitive and may be more likely to switch to a competitor if they find a better deal elsewhere. This can result in customers opting for lower-priced products or services, which can lower the average transaction value.')
    st.write('3. **Disengagement**: Customers who are at risk of churning may be less engaged with the company or brand and may be less likely to make high-value purchases. This reduced engagement can result in a lower average transaction value.')

    ### 
    # Points in Wallet

    st.write('## Points in Wallet')
    
    # points in wallet
    points_in_wallet = df_eda.groupby(['churn_risk_score']).aggregate(points_in_wallet=('points_in_wallet','mean')).reset_index()
    
    # plotting bar plot
    fig = px.bar(points_in_wallet, x="churn_risk_score", y="points_in_wallet",color='churn_risk_score',color_discrete_sequence=['red','blue'],
             orientation="v"       
                
             )
    fig.update_layout(title_text = "Points in Wallet")
    st.plotly_chart(fig)
    st.write('Based on visualization above, customers who are at risk of churning have a **lower points balance in their wallet** compared to customers who are not at risk of churning. Based on my assumption, customers who are at risk of churning may have a lower points balance in their wallet for several reasons:')
    st.write("1. **Reduced usage**: Customers who are at risk of churning may be using the company's products or services less frequently or may have stopped using them altogether. This reduced usage can result in a lower accumulation of points in their wallet.")
    st.write('2. **Disengagement**: Customers who are at risk of churning may be less engaged with the company or brand and may not be actively participating in loyalty programs or earning points. This reduced engagement can result in a lower accumulation of points in their wallet.')
    
    ###
    # Feedback

    st.write('## Feedback')

    # feedback
    feedback = df_eda.groupby(['churn_risk_score','feedback']).aggregate(Number_of_customer=('feedback','count')).reset_index()
    # plotting bar plot
    fig = px.bar(feedback, x="feedback", y="Number_of_customer",color='churn_risk_score',color_discrete_sequence=['red','blue'],
             orientation="v",hover_name="feedback"        
                
             )
    fig.update_layout(title_text = "Number of Customers Based on Their Feedback")
    st.plotly_chart(fig)
    st.write('Based on visualization above, The most feedback that causes customers to be at risk of churning is **poor product quality**. Based on my assumption, poor product quality can cause customers to be at risk of churning for several reasons:')
    st.write("1. **Reduced satisfaction**: Poor product quality can lead to reduced customer satisfaction, which can result in customers being less likely to continue using the company's products or services.")
    st.write("2. **Negative word-of-mouth**: Customers who experience poor product quality may share their negative experiences with others, resulting in negative word-of-mouth for the company. This can lead to a decrease in new customer acquisition and can also increase the likelihood of existing customers churning.")
    st.write("3. **Lack of trust**: Poor product quality can lead to a lack of trust in the company and its ability to provide high-quality products or services. This lack of trust can cause customers to be less loyal and more likely to switch to a competitor.")
    st.write("4. **Perceived value**: Poor product quality can result in customers perceiving less value in the company's products or services, which can make them less likely to continue using them and more likely to switch to a competitor.")

    ###
    # Correlation Matrix Analysis
    st.write('## Correlation Matrix Analysis')
    df_copy = df.copy()
    # Get Numerical Columns and Categorical Columns

    num_columns = df_copy.select_dtypes(include=np.number).columns.tolist()
    cat_columns = df_copy.select_dtypes(include=['object']).columns.tolist()

    # Using LabelEncoder to convert categorical into numerical data
    m_LabelEncoder = LabelEncoder()

    for col in df_copy[cat_columns]:
        df_copy[col]=m_LabelEncoder.fit_transform(df_copy[col])

    # Plotting Correlation Matrix of Categorical columns and default_payment
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(25,25))
    sns.heatmap(df_copy.corr(),annot=True,cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)

    st.write('Features that have have a strong correlation with the target variable (`churn_risk_score`) are `membership_category`,`points_in_wallet`,`avg_transaction_value`,`feedback`,`avg_frequency_login_days`,`joined_through_referral`,`preferred_offer_types`,`medium_of_operation`,`region_category` and `	offer_application_preference`.')

    ### 
    # Histogram and Boxplot based on user input
    st.write('## Histogram & Boxplot Based on User Input')
    select_column = st.selectbox('Select Column : ', ('age','days_since_last_login','avg_time_spent','avg_transaction_value','avg_frequency_login_days','points_in_wallet'))
    sns.set(font_scale=2)
    fig, ax = plt.subplots(1,2,figsize=(30,10))
    fig.suptitle(f'Histogram and Boxplot Visualization of {select_column} ')
    sns.histplot(ax=ax[0],data=df_eda[select_column],kde=True)
    ax[0].set_title(f'{select_column} skewness: {df_eda[select_column].skew()}')
    sns.boxplot(ax=ax[1],data=df_eda,x=df_eda[select_column],width=0.50)
    ax[1].set_title(select_column)
    
    st.pyplot(fig)
if __name__ == '__main__':
    run()