
from flask import Flask, request
import pandas as pd
from sys import argv
import sys
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import random
import os

app=Flask(__name__)
@app.route("/result",methods=["POST"]) 

def result():
    input_data=request.get_json() #get json data here
    merge3=pd.DataFrame(input_data)  # convertjson data into dataframe
    SpPaidClaims_RF = joblib.load('SpPaidClaims_RF.joblib')
    SpAmountPaid_ICD_RF = joblib.load('SpAmountPaid_ICD_RF.joblib')  
    SpAmountPaid_No_ICD_RF = joblib.load('SpAmountPaid_No_ICD_RF.joblib') 
    return_data=""
    try:
        ICD10_Zip=merge3[['Zip','ICD10']].drop_duplicates(['Zip','ICD10'])
        ICD10_Zip['ICD10'] = ICD10_Zip['ICD10'].str.replace('.','')
        ICD10_numeric=pd.read_csv('ICD10_1_Numeric.txt')
        ICD10_Zip1=pd.merge(ICD10_Zip,ICD10_numeric,on=['ICD10'],how='left') # Change ICD10 code to numeric
        ICD10_Zip1=ICD10_Zip1.drop(columns=['ICD10'])
        ICD10_Zip1=ICD10_Zip1.rename(columns={'ICD10_Numeric':'ICD10'})

        # expand relevant columns
        selected_df_total=pd.read_csv('df_total_selected3.csv')
        selected_df_total['ICD10']=pd.to_numeric(selected_df_total['ICD10'],errors='coerce')
            # merge
        ICD10_Zip2=pd.merge(ICD10_Zip1,selected_df_total,on=['Zip','ICD10'],how='left')
        ICD10_Zip3=ICD10_Zip2.drop(columns=['Unnamed: 0','Unnamed: 0.1'])

            # drop duplicates 
        ICD10_Zip4=ICD10_Zip3.drop_duplicates(subset={'Zip','ICD10'})

            # ## merge3 + ICD10_Zip4 (merging between Zia's external data and internal input data)
        merge4=merge3
        merge4['ICD10'] = merge4['ICD10'].str.replace('.','')
        merge5=pd.merge(merge4,ICD10_numeric,on=['ICD10'],how='left')
        merge5=merge5.drop(columns=['ICD10'])
        merge5=merge5.rename(columns={'ICD10_Numeric':'ICD10'})
        merge6=pd.merge(merge5,ICD10_Zip4,on=['Zip','ICD10'],how='left')

        provider_network=pd.read_csv('policyID_with_ProviderNetwork.csv')
        provider_network=provider_network.fillna(0)
        merge6=pd.merge(merge6,provider_network,on=['PlcPolicyId'],how='left')
        merge6_icd10=merge6[~merge6['ICD10'].isnull()]
        #merge6_icd10=merge6_icd10.drop(columns=['Unnamed: 0'])

        merge6_no_icd10=merge6[merge6['ICD10'].isnull()]
            #merge6_no_icd10=merge6_no_icd10.drop(columns=['Unnamed: 0'])

            # gender one hot encoding
        df_gender = pd.get_dummies(merge6['Gender'])
        df_new = pd.concat([merge6, df_gender], axis=1)
            # drop Gender
        df=df_new.drop(columns=['Gender'])
        merge6_SpPaidClaims=merge6[0:0] # initiate empty dataframe

        df=df.fillna(0) # change nan to zero
        for i in df['PlcPolicyId'].unique():  # go through unique PlcPolicyId
            df_PolicyId=df[df['PlcPolicyId']==i].mean()  # compute mean over the same PolicyId
            df_PolicyId=df_PolicyId.drop_duplicates()
            df_PolicyId=pd.DataFrame.from_dict(df_PolicyId).T
            merge6_SpPaidClaims=merge6_SpPaidClaims.append(df_PolicyId)

        SpPaidClaims_DF = merge6_SpPaidClaims[['PlcPolicyId','EffectiveDate','Gender','CalendarYearBirth','Zip','SIC','large_claims_counts','total_counts','ICD10']]
        merge6_SpPaidClaims=merge6_SpPaidClaims.drop(columns=['Gender','Zip','SIC','ICD10'])

        df=merge6_SpPaidClaims
        df=df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df=merge6_SpPaidClaims.fillna(0)

        Predicted_SpPaidClaims = SpPaidClaims_RF.predict(df)   # Add here other columns of SpPaidClaims for returning to user
        SpPaidClaims_DF['Predicted_SpPaidClaims']=Predicted_SpPaidClaims.round(1)
        SpPaidClaims_DF=SpPaidClaims_DF.drop(columns=['ICD10'])

        df1=merge6_icd10.fillna(0)

        SpAmount_paid_ICD1 = merge6_icd10[['PlcPolicyId','EffectiveDate','Gender','CalendarYearBirth','Zip','SIC','large_claims_counts','total_counts','ICD10']]

        df1_gender = pd.get_dummies(df1['Gender'])
        df1_new = pd.concat([df1, df1_gender], axis=1)
            # drop Gender
        df1=df1_new.drop(columns=['Gender'])
        Predicted_SpAmountPaid_ICD = SpAmountPaid_ICD_RF.predict(df1)   # Add here other columns of SpPaidClaims for returning to user


        SpAmount_paid_ICD1=SpAmount_paid_ICD1.rename(columns={'ICD10':'ICD10_Numeric'})
        SpAmount_paid_ICD=pd.merge(SpAmount_paid_ICD1,ICD10_numeric,on=['ICD10_Numeric'],how='left') # Change ICD10 code to alphanumeric
        SpAmount_paid_ICD=SpAmount_paid_ICD.drop(columns=['ICD10_Numeric'])
        SpAmount_paid_ICD['Predicted_SpAmountPaid']=Predicted_SpAmountPaid_ICD.round(1)

        df2=merge6_no_icd10.fillna(0)
        SpAmount_paid_no_ICD = merge6_no_icd10[['PlcPolicyId','EffectiveDate','Gender','CalendarYearBirth','Zip','SIC','large_claims_counts','total_counts','ICD10']]

            # drop string columns
        df2=df2.drop(columns=['FY 2019 FINAL Post-Acute DRG','FY 2019 FINAL Special Pay DRG','TYPE'])
            # gender one hot encoding
        df2_gender = pd.get_dummies(df2['Gender'])
        df2_new = pd.concat([df2, df2_gender], axis=1)
        df2=df2_new.drop(columns=['Gender'])

        Predicted_SpAmountPaid_No_ICD=SpAmountPaid_No_ICD_RF.predict(df2)# Add here other columns of SpPaidClaims for returning to user
        SpAmount_paid_no_ICD['Predicted_SpAmountPaid']=Predicted_SpAmountPaid_No_ICD.round(1)
        SpAmount_paid_DF = pd.concat([SpAmount_paid_ICD,SpAmount_paid_no_ICD],sort=False)

        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        random_str = ''.join((random.choice('abcdxyzpqr') for i in range(5)))

        SpPaidClaims_FileName='./Output/SpPaidClaims_'+random_str + date +'.csv' 
        SpAmountPaid_FileName='./Output/SpAmountPaid_'+random_str+ date + '.csv'
        SpPaidClaims_DF.to_csv(SpPaidClaims_FileName, index=False, sep=';')
        SpAmount_paid_DF.to_csv(SpAmountPaid_FileName, index=False, sep=';')
        return_data={"Path":"/Output","SpPaid_claim":SpPaidClaims_FileName,"SpAmoundPaid":SpAmountPaid_FileName}
    except Exception as er:
        return_data=str(er)

    return return_data

if __name__== '__main__':
    app.run(debug=True)