import pickle
import pandas as pd 
import numpy as np 


class HealthInsurance(object):
    
    def __init__(self):
        
        self.home_path = ''
        self.annual_premium_scaler         = pickle.load(open(self.home_path + 'features/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler                    = pickle.load(open(self.home_path + 'features/age_scaler.pkl', 'rb'))
        self.vintage_scaler                = pickle.load(open(self.home_path + 'features/age_scaler.pkl', 'rb'))
        self.region_code_encoder           = pickle.load(open(self.home_path + 'features/region_code_encoder.pkl', 'rb'))
        self.policy_sales_channel_encoder  = pickle.load(open(self.home_path + 'features/policy_sales_channel_encoder.pkl', 'rb'))
        
        
    def data_cleaning(self, data):

        # ordered columns
        data = data[['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']]
        
        # change columns names to lowercase
        data.columns = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage']
    
        # change region_code from float to int format
        data['region_code'] = data['region_code'].astype('int64')

        # change policy_sales_channel from float to int format
        data['policy_sales_channel'] = data['policy_sales_channel'].astype('int64')
        
        return data
    
    
    def feature_engineering(self, data):
        
        # change vehicle_age to easy interpretation
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else 
                                                                  'between_1_2_years' if x == '1-2 Year' else
                                                                  'below_1_year')

        # change vehicle_damage from object(yes & no) to int(1 & 0)
        data['vehicle_damage'] = data['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        return data
    
    def data_preparation(self, data):
    
        # annual_premium
        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values)
        
        # age
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)              

        # gender
        data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)            
        
        # vehicle age
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 0 if x == 'below_1_year' else
                                                                  1 if x == 'between_1_2_years' else
                                                                  2)         
        
        # region code
        data['region_code'] = data['region_code'].map(self.region_code_encoder)
                
        # Policy Sales Channel
        data['policy_sales_channel'] = data['policy_sales_channel'].map(self.policy_sales_channel_encoder)
        
        # select features 
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured']   
        
        return data[cols_selected]
    
    
    def get_prediction(self, model, original_data, test_data):
        
        # model prediction
        pred = model.predict_proba(test_data)
        
        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json(orient = 'records', date_format = 'iso')