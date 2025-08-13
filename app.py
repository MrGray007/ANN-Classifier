from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
model=load_model('model.h5')
with open('le_c.pkl','rb') as file:
    le_c=pk.load(file)
with open('le_g.pkl','rb') as file:
    le_g=pk.load(file)
with open('ohe.pkl','rb') as file:
    ohe=pk.load(file)
with open('ss.pkl','rb') as file:
    ss=pk.load(file)
def preprocessing(df):
    df_=df.copy()
    geo_ohe=(ohe.transform(df_[['Geography']]))
    gender_le=le_g.transform(df_[['Gender']])
    card=le_c.transform(df_[['Card Type']])
    df_['Gender']=gender_le
    df_['Card Type']=card
    df_=pd.concat([df_.drop('Geography',axis=1),geo_ohe],axis=1)
    df_=ss.transform(df_)
    return df_
st.title('Welcome')
geography=st.selectbox("Geography",ohe.categories_[0])
gender=st.selectbox('gender',le_g.classes_)
card=st.selectbox('Card Type',le_c.classes_)
age=st.slider('Age',18,92)
tenure=st.slider('Tenure',1,10)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
HasCrCard=st.selectbox('Has Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])
Complain=st.selectbox('Complain',[0,1])
Satisfaction_Score=st.slider('Satisfaction Score',1,5)
Points=st.slider('Points',100,1000)
NumOfProducts=st.slider('Num Of Products',1,4)
complain=st.selectbox('complain',[0,1])
input_df=pd.DataFrame(data={'CreditScore':[credit_score],'Gender':[gender],'Age':[age],'Tenure':[tenure],'Balance':[balance],'NumOfProducts':[NumOfProducts],'HasCrCard':
                            [HasCrCard], 'IsActiveMember':[is_active_member],'EstimatedSalary':[estimated_salary],'Complain':[complain],'Satisfaction Score':[Satisfaction_Score],'Card Type':[card],'Point Earned':[Points],'Geography':[geography]},)
input_df=preprocessing(input_df)
print(input_df)
pred=model.predict(input_df[0].reshape(1,-1))
print(pred)
st.write(f'Churn Probability {pred[0][0]}')
if pred[0][0]>0.5:
    st.write('Cutomer Not likely to churn')
else:
    st.write('Cutomer likely to churn')