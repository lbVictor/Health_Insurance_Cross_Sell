## INSURANCE ALL: A LEARNING TO RANK PROJECT
<img src="https://user-images.githubusercontent.com/85720162/153029328-5b4dfac7-b67d-4056-8344-fb7a21f8dcd9.png" width="1800">

------------------------------------------------------------------
## About the Project
For the development of this project, public data available in a Kaggle competition was used. From this data, a business context was created in which the company's product team is analyzing the possibility of offering policyholders a new product: A vehicle insurance. The objective of this project is to use the database provided by the company to develop a solution that consists of creating a purchase propensity score, for the sales team to identify which customers they should offer the vehicle insurance. To facilitate the usability of the solution, the availability of an ordered list with the customers most likely to purchase will be delivered initially and a spreadsheet on google sheets will be integrated into a machine learning model in production so that the sales team can generate the purchase propensity for future customers.

The methodology used to carry out this project was **CRISP-DM** which works through cyclical development with the objective of delivering value quickly.

**For the execution of this project the following tools were used:**
<p align="left">
 <table>
  <tbody>
    <tr valign="top">
      <td width="5%" align="center">
        <span>Python</span><br><br>
        <img height="40px" src="https://cdn.svgporn.com/logos/python.svg">
      </td>
      <td width="5%" align="center">
        <span>pandas</span><br><br>
        <img height="40px" src="https://pandas.pydata.org/static/img/pandas.svg">
      </td>
      <td width="5%" align="center">
        <span>NumPy</span><br><br>
        <img height="40px" src="https://numpy.org/images/logo.svg">
      </td>
      <td width="5%" align="center">
        <span>seaborn</span><br><br>
        <img height="40px" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg">
      </td>
      <td width="10%" align="center">
        <span>scikit-learn</span><br><br>
        <img height="40px" src="https://scikit-learn.org/stable/_images/scikit-learn-logo-notext.png">
      </td>
      <td width="5%" align="center">
        <span>LightGBM</span><br><br>
        <img height="40px" src="https://lightgbm.readthedocs.io/en/latest/_images/LightGBM_logo_black_text.svg">	    
      </td>
      <td width="5%" align="center">
        <span>Flask</span><br><br>
        <img height="40px" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png">
      </td>
      <td width="5%" align="center">
        <span>Heroku</span><br><br>
        <img height="40px" src="https://blog.4linux.com.br/wp-content/uploads/2018/01/Heroku.png">
       </td>
  </tbody>
</table>
</p>

Access the project development notebook here: 
https://github.com/lbVictor/Health_Insurance_Cross_Sell/blob/main/development/notebooks/health_insurance_cross_sell_cycle01.ipynb

## About the Company
Insurance all is an insurance company that offers health insurance to its policyholders and is evaluating the feasibility of starting to offer auto insurance to its current customers.

## Project Structure
### Business Understanding
00. **Business Problem:** Insurance All's product team wants to offer vehicle insurance to customers who already have health insurance. Last year they made a survey available to 380,000 customers asking if they would be interested in car insurance, and now based on that survey they want to identify in another database containing 127,000 customers what each customer's insurance purchase propensity is. To contact interested customers, the sales team is limited to 20,000 phone calls and the data science team was in charge of creating a solution that shows which customers are most interested in order to optimize service and generate greater results.

    For this project, a survey with customers was made available containing the features shown in the table below and data from 127,000 customers to apply the model and send to the service team.
	
  	**Features available in the dataset:**	  
	
   | Feature                 | Description |
   | ---                     | --- |
   | Id                      | Unique ID for the customer |
   | Gender                  | Gender of the customer |
   | Age                     | Age of the customer |
   | Driving_License         | 0: Customer does not have DL. 1: Customer already has DL |
   | Region_Code             | Unique code for the region of the customer |
   | Previously_Insured      | 0: Customer does not have auto insurance. 1: Customer already has auto insurance |
   | Vehicle_Age             | Age of the Vehicle |
   | Vehicle_Damage          | 1: Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past |
   | Annual_Premium          | Amount the customer paid the company for annual health insurance |
   | Policy_Sales_Channel    | Anonymous code for the customer contact channel |
   | Vintage                 | Number of days that the customer was associated with the company through the purchase of health insurance |
   | Response                | 0: The customer is not interested. 1: The customer is interested |
   
		
01. **Solution Plan:** After understanding the business problem, it was defined that a classifier with machine learning would be trained to generate the probability of each customer being interested in auto insurance and order a list of customers for the sales team to contact and make offers. For the sales team to generate the purchase propensity of future customers, we will upload the machine learning model into production and integrate it with a spreadsheet on google sheets for easier access.

### Data Understanding & Data Preparation

02. **Data Collection:** At this stage, the training data were collected through SQL in a Postgres database, stored inside an AWS machine. Production data were obtained by an API request.

03. **Data Cleaning & Data Description:** At this stage, the data were split between training and testing, the dimensions of the data were verified; data cleaning was performed: changing types to the correct format, analyzing if exists missing data, unique values was checked. A descriptive statistical analysis is also carried out to get an initial idea of the data and identify possible errors.

04. **Feature Engineering:** At this stage, a mind-map was created to model the phenomenon(vehicle insurance) and 10 business hypotheses were generated to be validated in the future. The way some features presented the values was modified to support the analysis and learning of the model, which will be carried out in the next phases.

05. **Variable Filtering:** At this stage, the objective is to analyze business limitations, data with wrong values or unnecessary columns to filter the dataset, but none of these changes were necessary.

06. **Exploratory Data Analysis (EDA):** At this stage, three types of analysis were performed in order to better understand the available data.
	* **Univariate analysis:** carried out in order to understand the individual behavior of each variable.
	* **Bivariate analysis:** carried out in order to understand the relationship of some features with the response variable through the validation of the raised hypotheses.
	* **Multivariate Analysis:** carried out in order to understand the relationship/correlation between all features + response variable.
	
07. **Data Preprocessing:** At this stage, the preparation of data for future application in machine learning algorithms was performed. The objective is to adjust the data without losing the information content in order to facilitate its understanding by machine learning algorithms.
	* **Rescaling:** For numerical variables, MinMax Scaler was applied in features without outliers ans Robust Scaler was applied in features with outliers.
	* **Encoding:** For categorical variables, Label Encoding, Ordinal Encoding and Frequency Encoding were applied according to the characteristics of each feature. 
		 
08. **Feature Selection:** In this step, feature importances from LightGBM algorithm and Random Forest were obtained to join with the knowledge obtained at EDA and choose which features would be used to perform the training of the models.


### Modeling
09. **Machine Learning:** In this step, the evaluation metric was defined; seven evaluation algorithms were trained (simple and sophisticated algorithms were applied to evaluate the results and verify the complexity of the phenomenon); cross-validation was performed to obtain the real results of the model; the hyperparameter fine tuning was performed to obtain the best parameters for the chosen model.
	
    - **Precision@k Metric**: The @k metrics are used when we want to apply a metric limiting the examples to a value (k). After generating the purchase propensity, the dataset was sorted from the customer with the highest propensity to the lowest and I used as a value k the amount of people interested in the dataset in which I was applying the model (validation or testing), so customers who are up to that value are the customers that my model predicted to be most interested in vehicle insurance. A precision @k of 100% means that all of my model's predictions were right, 50% means that up to the @k value my model was right 50% of the time, etc.
  
  
    - **Machine Learning Results with Cross-Validation and the size of the trained model:**
	
      ![image](https://user-images.githubusercontent.com/85720162/153069641-8fa6c186-ad43-4dfc-898d-a31385b31a08.png)
      
      Based on the results obtained by the models, **LightGBM** was chosen because it presented a very satisfactory result in relation to the others and the size of the trained model is the smallest among those with good results.

    - **Hyperparameter Fine Tuning:** Bayesian Optimization was used to obtain the best parameters for the chosen model.
	
	    **LightGBM Final Result:**
      
	    ![image](https://user-images.githubusercontent.com/85720162/153072721-58df50a6-ab99-4964-a4f7-1a036d8736de.png)

	
###	Evaluation

10. **Performance:** The performance of the model was evaluated from a Machine Learning perspective and from a business perspective to verify the final result of the model and its impact on the business.

  * **Machine Learning Performance:**

    - **Precision@k:**
  
      ![image](https://user-images.githubusercontent.com/85720162/153074144-df928a95-e1b9-46ae-b25e-91b135224695.png)


    - **Gain & Lift Curve:**
  
      <img src="https://user-images.githubusercontent.com/85720162/153074311-a6809e58-39a2-4701-a51c-b36bd8c1b883.png" width="700">
      
      As we can see in the precision@k and lift curve above, there was a small difference between the validation and test results, however, the result under the test data was 2.8x greater than the random result (without the propensity to purchase score, represented by the black line).
		
* **Business Performance:** 
  
  - **Financial Results:**
  
  - **business questions answered:**
    1. Percentage of customers interested in vehicle insurance in 20,000 calls
    2. Percentage of customers interested in vehicle insurance in 40,000 calls
    3. Number of calls needed to contact 80% of those interested in vehicle insurance
	


###	Deploy
11. **Deploy Model to Production:** To upload the model into production, an application was created on Heroku Cloud containing an API with the entire pipeline of transformations necessary for the raw data to be eligible for the application of the stored model, which receives the data and returns the purchase propensity. To facilitate future forecasts by the sales team, a spreadsheet was created in Google Sheets integrated into the model, where it is possible to obtain the purchase propensity just by clicking on a button created through Google Scripts.


<p align="center">
Google Sheets Application 
</p>

<p align="center">
 	<img width="800" alt="drawing" src="https://user-images.githubusercontent.com/85720162/153077216-b2995c43-d0e0-424e-bda8-e0cb2441e17c.jpg">
</p>

<p align="center">
To access the spreadsheet and perform the prediction click here
</p>

* **Next Steps:** For the next CRISP cycle, we can make several improvements to the project:

	- 
	- 
	- 
	- 

* **Learns:** 
