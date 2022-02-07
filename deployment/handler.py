import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# load model
model = pickle.load(open('models/model_lightgbm.pkl', 'rb'))


# Initialize API
app = Flask(__name__)

# create a route
@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    
    # get data
    test_json = request.get_json()
    
    # there is data
    if test_json:
        
        # unique example
        if isinstance(test_json, dict): 
            test_raw = pd.DataFrame(test_json, index=[0])
            
        # multiple examples
        else: 
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Health Insurance class
        pipeline = HealthInsurance()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    
    # there isn't data
    else:
        return Response('{}', status=200, mimetype='application/json')
    
# run API
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)