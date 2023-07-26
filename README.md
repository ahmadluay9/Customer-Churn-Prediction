# Customer Churn Prediction Using Artificial Neural Network in E-commerce Company

This project aims to build an Artificial Neural Network model that capable of predicting customer churn in an e-commerce company. By predicting customer churn, a company can take proactive measures to retain these customer

# File Explanation on Github

This repository consists of several files, namely :

- **Folder deployment** = Contains files used for deployment to HuggingFace (contains models, python applications etc.)
- **churn.csv** = dataset used for this project
- **Notebook_Customer_Churn_Prediction.ipynb** = This file is the main notebook used to explore dataset and built model
- **model_inference.ipynb** = - Notebook used for testing inference. Inferencing is done on a separate notebook to prove that the model can run on a notebook that is clean of variables
- **h8dsft_p2m1_Ahmad Luay Adnani_inference.csv** = Dataset used for Model Inference
- **model_seq2.h5** = Saved model to be loaded and used later for making predictions on new data without the need to retrain the model.
- [**Deployment**](https://huggingface.co/spaces/ahmadluay/Customer_Churn_Prediction)

# Brief Summary of Project

The flow of this project, first EDA (Exploratory Data Analysis) to find out the basic picture of the dataset. Second, cleaning and preprocessing of the dataset. Third, Built 4 Model (Sequential, Improved Sequential, Functional, Improved Functional) and choose Improved Sequential as Best Model. 

# Project Conclusion
- The selected model is the sequential model that has been improved with model improvement. This model also has the highest recall for label 0 (customers who are at risk of churning) and has the lowest false negative (customer is not at risk of churning predicted as customers is at risk of churning).

- To retain e-commerce customers who are at risk of churning, it is important to address the factors that contribute to their likelihood of churning. Some effective strategies may include offering loyalty rewards programs, providing personalized offers and promotions, improving the overall customer experience, and investing in customer engagement through targeted email campaigns and social media outreach.
