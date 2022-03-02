#!/usr/bin/env python
# coding: utf-8

# <br>
# <h1 align="center">Customer Churn Report - Building an ML model</h1><br>
# 
# <h1>Context:</h1>
# <br>
# 
# <br> On 2021, the <b>average churn rate in telecom businesses was 22%</b>. Every year, it has been increasing mainly because of a <b>strong competition</b>. 
# <br>The telecommunication industry is composed of a large variety of service providers and customers just change form one to one easily.
# Individual customer retention is difficult because most telecom companies have too many customers that they cannot devote more time to than necessary. 
# 
# <br>Our agency has been hired by a Telecom company  as an AI expert to build a model that will enable them to predict the customers with a high churn probability based on a <b>2-years historic customers data</b>. 
# They would like to put in place strategies to retain customers with high probability of churn in order to concentrate their efforts and minimize their expenses. 
# 
# <br>
# <h1>Problem definition </h1>
# <h3>What is our topic ?</h3>
# 
# <br> Our goal is to enable the telecom company to predict the customers who have a high probability of churn with a good precision to help them activate the right retention strategies.
# <br>
# Customer churn is one of the <b>most important metrics</b> a growing business needs to evaluate. 
# 
# <b>Definition</b><br>
# <i>"Customer churn is the percentage of customers that stopped using your company's product or service during a certain time frame"</i> (Hubspot). 
# 
# 
# It helps the company to identify customers who are going to churn and understand the reasons behind it, harnessed by the power of data and machine learning. And it enables the adequate teams and allow them to develop the tactics to achieve <b>customer retention</b>. 
# 
# <br>
# <h3>Our methodology</h3>
# 
# We have a database with demographic and account related information about the customers and the services they subscribed to in the telecommunication company. <br>The identified problem has customer input variables and an ouput variable, which is our <b>target : "the customer churn"</b>. It is <b>a supervised learning model</b> that is needed to learn from the data and be used to make churn predictions.
# <br>
# First, I will <b>clean and transform this dataset</b>. I will explore our data by visualizing and understanding it to know which information will be the most useful for our analysis and if some adjustements are needed (outliers ? categories...etc).
# Then, I will evaluate the following <b>prediction models</b> : 
# - logistic regression
# - random forests
# - support vector machines
# - Decision Tree Regressor
# 
# Afterwards, I will <b>test each model</b> and evaluate its <b>performance on Churn predictions</b>. <br>
# Finally, I will <b>conclude by selecting the best model fine-tuned with hyperparameters.</b>
# <br>
# <h3>Objectives</h3>
# 
# I will focus my report and my analysis on the following research questions:
# - Can we predict the customers who are going to churn using a supervised learning model ?
# - Are there features that are correlated to customer churn ?
# - Are there different types of churn (satisfaction ? product/service related ?) ?
# - What can we recommend the Telecom company to do ?
# 

# In[131]:


import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd

#Splitting our traing & test set using stratified sampling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#Build a pipeline :
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#Building ML models, using cross-validation and analyzing models performances:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


url = 'https://bit.ly/telecom_customer_churn'
if True:
    df = pd.read_csv(url)
else:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv.xls')

df.shape


# # 1. Data collection :

# From the Dataset descriptive information, we already know that there are <b>7043 rows</b> corresponding to customerID and <b>21 columns or features</b>. 
# The columns are describing customers account information, demographic information, customer subscribed services and the customer churn Vs previous month:
# 
# <li> The <b>Churn column</b> is binary with Yes/No values for each customer: 
# 'Yes' representing a customer who churned i.e he left the company within the last month. 'No' corresponding to a current customer of the telecom company. It is our <b>target</b>.</li>
# 
# <li> Other columns represent <b>customer account information</b> : tenure (duration in the company), customer payment method, payperless billing (yes or no), subscription or contract type (month-on-month, one year-contract or 2-years contract) and monthly and annually total charges.</li>
# 
# <li> Additonal columns represent the <b>customer subscribed service (or not) </b>: phone service, multiple lines, internet service, online security, device protection, technical support, TV streaming and movies streaming.</li>
# 
# <li> And finally; there are the columns representing the <b> demographic information for each customer</b> : gender type, if they are senior citizens and if they have a partner and dependents.</li>

# # 2. Data Preprocessing :

# <h2> 2.1 Data Cleaning </h2>

# In[132]:


df.head()


# In[133]:


df.info()


# In[134]:


#converting TotalCharges into numeric column:
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[135]:


df.isna().sum()


# In[136]:


df.loc[df['TotalCharges'].isna()]


# The Dataset has 7043 rows (corresponding to a unique customer), 21 columns (features).
# The dataset is composed of 2 integer, 2 float and 17 object (categorical) types columns. 
# But when we look closer, it appears that the TotalCharges column is composed of decimal values but its type is an object. It has been converted into float. And we identify 11 NaN values which should be dealt with. Even if they are corresponding to loyal customers who did not churn (one to 2 years contracts), we cannot keep them because TotalCharges != (MonthlyCharges * Contract duration).
# 
# The best way to deal with this missing Data (and avoid having bias in our model) is to drop these rows.

# In[137]:


#supressing the 11 rows with NaN values for TotalCharges:
df = df.copy().dropna()


# In[138]:


df.info()


# Now, TotalCharges has been converted to float and there is a total of 7032 rows. 
# I will check if any customer data is duplicated or if a customer is appearing several times with his customerID.

# In[139]:


df[df['customerID'].duplicated()].any()


# There is no duplicated customerID. 

# <h2> 2.2 Data exploration </h2>

# In[140]:


# target variable :
churn_label = df['Churn']


# ###### Churn proportion : 

# In[141]:


churn_prop = round(df.groupby('Churn').size()/len(df)*100,1)
churn_prop


# There is a <b>churn rate of around 27% </b> of the customers Vs previous month. 
# It is normal to have a churn rate or customers who disrupt the service, even though it is slightly above the market's (22% in 2021). 
# 
# The objective is to provide the Telecom company with a direction for its services improvements and to enable them identify churning customers by predicting churn. The telecom company will thus be able to investigate the churn reasons with the customers and activate a retention strategy through the concerned departments (according to the features identified during the exploratory phase).
# 
# As a first step, to understand our Dataset, I will visualize the numeric and the categorical variables then I will convert the categorical Data into numerical to prepare the dataset for the maching learning model.

# In[142]:


#defining the categorical features and dropping the Customer Id as it makes no sense to keep it for predictions :
cats = list(col for col in df.columns if (df[col].dtype == 'object') & (col not in ['customerID']))
cats 


# <h3> Treating categorical variables, understanding the Data and the correlations: </h3>

# Let's have a look at our features categories to understand and treat them :

# In[143]:


d_cats = df[cats].copy()
d_cats.head()


# In[144]:


def feature_cat(dataframe):
    for col in dataframe:
        print('{} : {}'.format(col, dataframe[col].unique()))
feature_cat(d_cats)


# Some columns contain values with the <b>same meaning but are not grouped</b> :
# 
# MultipleLines : 'No phone service' means simply 'No' and should be replace by 'No'
# OnlineSecurity : 'No internet service' should be 'No'
# OnlineBackup : 'No internet service' should be 'No'
# DeviceProtection : 'No internet service' should be 'No'
# TechSupport :'No internet service' should be 'No'
# StreamingTV : 'No internet service' should be 'No'
# StreamingMovies :'No internet service' should be 'No'
# 
# I will correct the categories or values for these columns by simply <b> replacing all 'No internet service' and 'No phone service'</b> by <b>'No'</b>

# In[145]:


d_cats = d_cats.replace('No internet service', 'No') 
d_cats = d_cats.replace('No phone service','No')


# Sanity check :

# In[146]:


feature_cat(d_cats)


# <h3> Data Visualization & exploring the correlations </h3>

# <h4><li> For the categorical Variables </li></h4>
# To compare categorical variables, I use below bar charts to visualize the relationship between them (or not)

# In[147]:


import seaborn as sns

for var in cats:
    sns.countplot(data=d_cats, x = var, hue = churn_label)
    plt.title(var)
    plt.tight_layout()
    plt.show()


# The comments below are made on visual observations only. They should be taken into the recommendation stage to the company only after building our machine learning model and confirming the features correlations with the churn rates.<br>
# 
# <b> Below are the features which look enfavoring churn and which would need specific further investigation in a phase 2 and/or by customer service teams:</b>
# <li><b>MultipleLines</b> : <br>comparing the proportions of the bars, it looks at a first glance that the <b>customers with multiple lines are churning more than those who don't</b>. And probably even more that the average churn rate. The reason behind this would probably need to be investigated</li>
# <li><b>The Internet service with the optic fiber</b>: <br> It seems to be a factor prone to churn as the customers who churn are barely (a little bit lower) equal to those who don't. This service is <b>above the expected average churn rate</b>. <br><li><b>It is the same with Streaming services and Streaming Movies services</b>, where we can notice that there are more customers using the services who churn than those who don't.</li>
# <br>The Telecom company would need to <b>dig a little deeper into the numbers</b> and if confirmed, <b>recommend the customer service responsible for each product (or service) and the marketing teams to make a follow up on the customer journey and put the right tactics in place to adjust their operations according to the customers'expectations and needs.</b></li>
# <br><br>
# <b>Some other features characteristics listed below enfavor customer retention :</b>
# <li><b>Online security, online-backup & Tech support </b>: <br>It seems that the <b>customers with online security, an online back-up and Tech Support are less prone to churn</b> and even below the average churn rate than those who don't. And for Tech Support, it even looks like this is a service that not only decreases churn rate but also when it is not subscribed, it is a service that increases churn.
# <li>Finally the <b>automatic payment methods seem to be factors that prevent customers from churning</b> : 'Bank transfer (automatic)','Credit card (automatic)'. Regarding mailed check it looks to be also a predictor of loyal customers Vs churning customers. Whereas electronic check seems to be related to Churn. These observations can be further explored in another stage to find some attributes combinations.</li>
#     
# <br><b>Nothing specific to declare :</b>
# <li>For gender, there is no distribution difference between males and females. And it looks like the proportions of churn are respected : about 1/3 churn for each group, which is approximately the churn rate we found above (27%)</li>
# <li>For dependents, it looks like the customers who don't have dependents churn more (50% Vs 25% for those who have dependents.</li>
# <li>Very few customers don't have phone services. Overall the churn rate seems respected (those who churn represent of those who stayed)</li>
# <li>Customers with Paperless Billing seem to be more likely to churn that those who don't. The reason behind this is probably behavioral more than linked to the PaperlessBilling. This can be investigated by the telcom company with surveys.</li>

# <h4><li> For the numeric Variables </li></h4>

# In[148]:


#defining the numerical features :
nums = list(df.select_dtypes('number').columns)

#Let's have a look at the summary and the distribution of our numeric data :
df[nums].describe()


# <li><h4> Plotting Histograms for numeric attributes </h4></li>

# In[149]:


#transforming churn into a numeric feature:
list_churn = df['Churn'].map({'Yes' : 1, 'No': 0})
df_churn_num = pd.DataFrame(list_churn)

df_nums = pd.merge(df[nums],df_churn_num,left_index = True, right_index =True)

df_nums.head()
df_nums.head()


# In[150]:


df_nums.hist(figsize =(15,15), bins = 50)
plt.show()


# According to the tenure histograms distribution, customers are well distributed across tenure and there are two spikes visible : 25% are under a tenure of 9.0 (acquired customers) and 25% are above 55.0 (loyal customers); the <b>remaining 50%</b> (!) are varying between 9.0 and 55.0, meaning that they are <b>still satisfied by the services provided...until some point in time or a trigerring event.</b> 
# 
# At some point in between, the Telecom company loses 25% of customers who could have stayed some more months and who decided to disrupt the services and quit the company. 
# 
# <b>The ML model will determine the features which will help the Telecom company identify these churners and support them in </b>identifying (then investigating on a one-on-one basis potential churn reasons and deploying) the right sales, marketing or customer support strategy to retain these customers.</b>
# <br>
# <br>
# I also notice that there is a shape of a <b>fat tailed distribution</b> and <b>75% of the customers Total Charges are below 4000</b> in local currency. But 25% of the customers are above, some of them reaching total charges of about 8000 in local currency.
# 
# This raises an important step in the data preparation to build the machine learning model : <b>some total charges go far beyond 4000 !!</b> 
# This is the reason why, I will handle the category above 4000 as one category in a step below.
# <br> It is important to have a <b>sufficient number of instances in the dataset for each stratum</b> in order to have a good <b>representation of all the customers'segments</b> otherwise the <b>estimation of the most profitable and newly acquired customers will be biased</b>. 
# This means that I will need to limit the number of Total charges categories. 
# I will divide the total charges attribute by 2266 corresponding to the standard deviation (to limit the number of TotalCharges categories), and I will round it up by using ceil and merge the categories greater than 4000 into category 4000.
#     
# <br> The company will certainly need to <b>dig more into their data and identify these highly profitable 25% of its customers</b>, providing them with different quality of service to retain the most profitable ones. <b>A customer segmentation analysis is recommended</b> to be developed as a <b>second phase analysis</b> (it is not the subject of this research) to <b>fine-tune the customer relationship strategy of the company</b> and adapt the strategy to each customer segment.

# <li><h4> Searching for correlation : </h4></li> 
# In order to identify correlation within the data, I will transform the target variable (Churn) into numeric data.
# Secondly I will draw a <b>heatmap using the kendall method to observe the data correlation</b>. This method enables to measure the ordinal association between the variables :

# In[151]:


ax = sns.heatmap(df_nums.corr(method='kendall'), annot=True, fmt='.2f',vmin=-1, vmax=1, cmap='coolwarm')
plt.tight_layout()
plt.show()


# <li><h4> Looking for correlation </h4></li>

# In[152]:


df_nums.corr()


# With no surprise, the total charges and the monthly charges are correlated (0.65) and Total charges are correlated with tenure (0.82). <br><b>The numeric variables correlated with churn are:</b>
# <li>Tenure : negatively correlated, stronger correlation (-0.35)</li>
# <li>Total Charges: second stronger correlation, negatively correlated (around -0.2)</li>
# <li>Monthly charges and Senior citizen are positively correlated.</li>
#     
# The relationship between <b>totalcharges and tenure being negatively correlated with churn</b> is interesting, because it reminds us of the customer lifetime value (CLV = frequency * average basket * length of the relationship). 
# <br> A second phase analysis to segment the customers with their CLV would be recommended to the Telecom company so that it can target specifically and invest more on the retention of the customers with a high CLV.

# # 3. Data preparation for ML algorithm :

# <h1> 3.1 Creating a test set :</h1>

# <li><h2>Rebuilding our dataset :</li></h2>
# I recreate here the full dataset with the numeric categorical features as per the data cleaning and preprocessing steps. 

# In[153]:


#sanity check to make the merge :
len(d_cats) == len(df_nums)


# In[154]:


#let's rebuild the full dataframe with the categorical and numeric features.
#The rows correspond to the same customerID so inner join will do the job:

df_full = d_cats.merge(df_nums, how = 'inner', right_index=True, left_index=True)

#dropping the column "Churn" (categorical from d_cats) and renaming Churn_y into Churn
df_full = df_full.drop('Churn_x', axis = 1).rename({'Churn_y': 'Churn'}, axis =1)

df_full.head()


# <li><h2> Using random selection to create a test split </h2></li>

# As seen above, the TotalCharges categories need to be limited and ceiled. This is confirmed by the value_counts below :

# In[155]:


df_full['TotalCharges'].value_counts()


# Customers are too largely distributed...many of them are alone. And indeed, we have a fat tail distribution.

# In[156]:


plt.hist(df_full['TotalCharges'])


# The Pareto Law looks totally in action here. An interesting finding would be to see how it is applied and to search for our 20% most profitable customers (generating 80% of our revenues), the services they subscribed for and/or characteristics and behaviors...

# In[157]:


# Dividing the Total charges by 2266 (standard deviation) to limit the number of TotalCharges categories 
#And let us understand the distribution 
df_full["TotalCharges"] = np.ceil(df_full["TotalCharges"] / 2266)

# Label those above 4000 (75% of our data) as 4000
df_full["TotalCharges"].where(df_full["TotalCharges"] < 4000, 4000, inplace=True)


# In[158]:


df_full["TotalCharges"].value_counts()


# In[159]:


plt.hist(df_full['TotalCharges'])


# This looks better categorization to analyze our customers'segments characteristics.

# <h4><li>Using Stratified selection </li></h4>

# Now, I will use Scikit-Learn’s StratifiedShuffleSplit to do a stratified sampling and be able to have a representativity of the different categories within the test and the sample datasets.

# In[160]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_full, df_full["Churn"]):
    strat_train_set = df_full.iloc[train_index]
    strat_test_set = df_full.iloc[test_index]
strat_test_set


# In[161]:


X_train = strat_train_set.copy().drop(labels = ['Churn'],axis = 1)
y_train = strat_train_set['Churn'].copy()

X_test = strat_test_set.copy().drop(labels = ['Churn'],axis = 1)
y_test = strat_test_set['Churn'].copy()


# <h4><li>Building a pipeline to preprocess the categorical and numeric input features :</li></h4>

# In[162]:


#defining the list of numeric and categorical attributes for which columns we will apply 
#the transformations through the pipeline

num_attribs = list(X_train.select_dtypes('number').columns)
cat_attribs = list(set(X_train.columns) - set(num_attribs))

#we need to apply transformations into a single pipeline with standardization of the numeric values
#and transformation of the categorical values :

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# As Machine Learning algorithms don't perform well on numeric features with different scales, I will standardize the numeric features. 
# For this, I apply a standardization which will substract the mean value then dividing by the variance resulting in a distribution with unit variance. The numeric values will less be impacted by outliers.

# In[163]:


#Building a pipeline for numeric attributes to standardize the values :

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

#Building the full_pipeline :
#For the categorical values transformation,we use directly the embedded OneHotEncoder from Scikit-Learn 
#OneHotEncoder converts integer categorical values into one-hot vectors (binary).

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])


# In[164]:


X_train_prepared = full_pipeline.fit_transform(X_train)


# In[165]:


X_train_prepared


# In[166]:


X_train_prepared.shape


# In[167]:


X_test_prepared = full_pipeline.fit_transform(X_test)


# I will use a second pipeline both for preprocessing the features with columTransformer (full_pipeline mentionned above) and to build the models.
# <br>
# To <b>ensure the future performance of the models and avoid data leakage, the pipeline will be passed at every step of the workflow </b>: 
# from each fold of the cross-validations directly on the training and the test sets (not transformed), GridSearchcv and preprocessing with hyper parameters. 

# # 4. Selecting and training a model

# <h2>Training and evaluating on the training set & visualizing the models performance and interpreting the results </h2>

# <h3>Stratified cross-validation</h3>

# In order not to waste too much data of our dataset and contaminate it, risking to overfit our model. I will use a <b>10-fold cross-validation</b> (and a 5-fold cross-validation for GridSearchcv with SVC model), trying to find a <b>trade-off between the variance and the bias in our model selection</b>. 
# Then, I will search for the best predictive model for Churn using our dataset.
# <br>
# Every time, the pipeline fits the models right after transforming the Data in order to avoid contaminating it as much as possible.

# <h3><li> Logistic Regression model </h3></li> 

# In[168]:


from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

lr_model = Pipeline(steps=[('preprocessor', full_pipeline),
                      ('classifier', LogisticRegression(solver="lbfgs", 
                        max_iter=300, class_weight="balanced"))])

scores = cross_val_score(lr_model, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(scores )


# In[169]:


final_model = lr_model.fit(X_train, y_train)


# In[170]:


round(lr_model.score(X_train, y_train),2)


# In[171]:


round(lr_model.score(X_test, y_test),2)


# In[172]:


y_predict_lr = lr_model.predict(X_test)
print(y_predict_lr)


# In[173]:


from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_test, y_predict_lr)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# <h4>Interpretation of the logistic regression results :<h/4>

# The <b>scores are equivalent between the 10-folds</b> and the <b>mean score is about -0.25 and the RMSE is 0.52 (could be better)</b>.
# <br>
# The training score and the testing scores are equivalent, respectively 0.76 and 0.73, which is good.

# <h3><li> SVM - Support Vector Machine </h3></li> 

# In[174]:


from sklearn.svm import SVC

svc_model = Pipeline(steps=[('preprocessor', full_pipeline), ('classifier', SVC(random_state = 1))])

scores = cross_val_score(svc_model, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

svc_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(svc_rmse_scores )


# In[175]:


final_svc_model = svc_model.fit(X_train, y_train)


# In[176]:


#training score :
print('training score is :', round(svc_model.score(X_train, y_train),2))


# In[177]:


#Making predictions :
y_predict_svc = final_svc_model.predict(X_test)


# In[178]:


#Measuring SVC predictions accuracy :
accuracy_svc = final_svc_model.score(X_test,y_test)
print("SVC accuracy:",accuracy_svc)


# In[179]:


#Classification report for the SVC model:

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict_svc))


# <h4>Interpretation of the SVM model results :<h/4>

# With the SVC model, the standard deviation of 0.013 and the mean RMSE score across cross validations is of 0.44. 
# <br>The training score is 0.82 Vs 0.79 for the testing score. The model <b>generalizes well to new data</b> (test data).
# <br> The training accuracy is 0.79. <b> So the performance of the SVC is good.</b><br>
# 
# <b>Classification report analysis :</b>
# <br>Now, let's have a look at the classification report to check the business application of this model for the telecom company.  <br>
# The <b>recall is 48%, thus 52% of the churners are not identified.</b> 
# AND the <b>precision is about 63% meaning that if churners are identified, only one third are false positives.</b> It is acceptable from expenses prospective as two third of churners are identified and a strategy can be put in place to retain them.<br>
# In other words, the company can use this model to predict churn and recommend to develop retention strategies by the customer/sales or marketing services.

# <h3><li> Decision Tree Regressor </h3></li> 

# In[180]:


from sklearn.tree import DecisionTreeRegressor

tree_reg_model = Pipeline(steps=[('preprocessor', full_pipeline), ('classifier', DecisionTreeRegressor())])

scores = cross_val_score(tree_reg_model, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

tree_reg_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_reg_rmse_scores )


# In[181]:


final_tree_reg_model = tree_reg_model.fit(X_train, y_train)


# In[182]:


round(tree_reg_model.score(X_train, y_train),2)


# In[183]:


#Making predictions :
y_predict_tree_reg = tree_reg_model.predict(X_test)


# In[184]:


round(tree_reg_model.score(X_test, y_test),2)


# The training score is 0.99 with an RMSE score across each fold remains constant and is of 0.52 with a standard deviation of 0.024. 
# But the test score is catastrophic with -0.41. The model is clearly <b>overfitting</b> the training data.

# <h4>Interpretation of the decision tree regressor results :<h/4>

# The model overfits.
# AND we <b>WILL NOT USE this model to make churn predictions</b> and business recommendations.

# <h3><li> Random Forest model </h3></li> 

# In[185]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics

rf_model = Pipeline(steps=[('preprocessor', full_pipeline), ('classifier', RandomForestClassifier(n_estimators=500 , 
                                                            oob_score = True, 
                                                            n_jobs = -1, 
                                                            random_state =50, 
                                                            max_features = "auto",
                                                            max_leaf_nodes = 30))])

scores = cross_val_score(rf_model, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(forest_rmse_scores )


# In[186]:


final_model = rf_model.fit(X_train, y_train)


# In[187]:


print('The training score of the random forest model is : ', round(rf_model.score(X_train, y_train),2))


# In[188]:


print('The testing score of the random forest model is : ', round(rf_model.score(X_test, y_test),2))


# In[189]:


# Make predictions
from sklearn.datasets import make_classification
from sklearn import metrics

y_predict_rf = rf_model.predict(X_test)
print ('Accuracy score of the random forest is :', round(metrics.accuracy_score(y_test, y_predict_rf),3))


# In[190]:


#evaluating the accuracy
print(classification_report(y_test, y_predict_rf))


# In[191]:


#Confusion matrix with random forest :
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_predict_rf),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()


# In[192]:


#plotting the ROC curve for random forest to show the trade-off between sensitivity (true positives) and the specificity (false positives):

from sklearn.metrics import roc_curve

y_rfpred_prob = rf_model.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_rf, tpr_rf, label='Random Forest',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show();


# <h4>Interpreation & estimation of the random forest predictions :<h/4>

# With the random forest model, the training socre and the test score are very close and good, respectively 0.82 and 0.79. 
# The<b> accuracy score is of 0.795, almost equal to the one obtained with the SVC (0.79)</b> and <b>it is close to 1, which is a pretty good performance</b>.
# 
# The RMSE score remains the same between the folds during the cross validation and is equal to the one obtained with the SVC, with a <b>mean score of 0.44 </b> + a standard deviation of 0.018.<br>
# <br><b>Classification report :</b>
# <br>
# Let's analyse the classification report to consider if it can be used for a business application (or not).
# The SVC was pretty good to predict with a precision of 0.63 and a recall of 0.48.<br>
# For the random forest, the classification report shows a precision of 0.66 and a recall of 0.45</b> so the model is performing as good as the SVC model for both the precision and the recall with probably a little more false negative but less false positive. 
# <br>We cannot really decide to discreminate between one or the other at this stage, this is the reason why we will be running the <b>GridSearch to fine-tune both models and find the best hyperparameters before comparing their performance and deciding which one is better.

# # 5. Fine-tuning the models

# <h3>5.1 GridSearchCV for SVM model</h3>
# 
# Let's start with the SVM model and apply the <b>Grid Search cross validation to select the best hyperparameters for the model</b>.

# In[210]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svc_model = SVC(random_state = 1)

parameters = {'C':[1,10,100],'gamma':[1,0.1,0.001], 'kernel':['linear','rbf']}

grid_search = GridSearchCV(estimator = svc_model,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 5, verbose =2)

grid_search.fit(X_train_prepared, y_train)


# In[211]:


#displaying the best hyperparameter combination :
print('\nThe Best Parameters of our Given Model are:', grid_search.best_params_)


# In[212]:


#displaying the best estimator :
print('best estimator : ', grid_search.best_estimator_)


# In[213]:


#Displaying the score of each hyperparameter combination tested during the grid search:
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print('RMSE score : {} for {}'.format(np.sqrt(-mean_score), params))


# In[214]:


#We define below the name of our best model :
best_svc_model = Pipeline(steps=[('preprocessor', full_pipeline), ('classifier', grid_search.best_estimator_)])


# <h3>5.2 GridSearchCV for Random Forest model</h3>
# 
# Let's now fine-tune the Random Forest model and apply the <b>Grid Search cross validation to select the best hyperparameters for the model</b>.

# In[220]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2,3,4]},
]

forest_reg = RandomForestRegressor()

grid_search_rf = GridSearchCV(forest_reg, param_grid, cv =5, 
                           scoring = 'neg_mean_squared_error',
                           return_train_score = True)

grid_search_rf.fit(X_train_prepared, y_train)


# In[221]:


#displaying the best hyperparameter combination :
grid_search_rf.best_params_


# In[222]:


#displaying the best estimator :
print('best estimator : ',grid_search_rf.best_estimator_)


# In[223]:


#Displaying the score of each hyperparameter combination tested during the grid search:
cvres = grid_search_rf.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print('RMSE score : {} for {}'.format(np.sqrt(-mean_score), params))


# In[198]:


#We define below the name of our best model :
best_rf_model = Pipeline(steps=[('preprocessor', full_pipeline), ('classifier', grid_search.best_estimator_)])


# <h3>Analyzing the best models, their errors and selecting the best model: </h3>

# For the the Random Forest model, below is the relative importance of each attribute for making accurate predictions :

# In[224]:


feature_importances = grid_search_rf.best_estimator_.feature_importances_
feature_importances


# In[225]:


cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# The <b>most important attributes for churn prediction confirmed with the random forest final model are tenure and monthly charges</b>.

# # 6. Evaluating our best estimator on the test Set and conclusion

# <h3>6.1  SVC model</h3>

# In[215]:


#predictions using the SVC best model :
best_svc_predictions = best_svc_model.predict(X_test)
best_svc_predictions


# In[229]:


#evaluating the SVC best model performance :
best_svc_model.score(X_test, y_test)


# In[217]:


RMSE_best_model = mean_squared_error(y_test, best_svc_predictions, squared = False)
RMSE_best_model


# In[218]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,best_svc_predictions)


# The score is very good (close to 1) and is consistant between the training score and the test score, demonstrating that the model generalizes well to new data and that it will do well in predicting churn for the telecom company.

# <h3>Random Forest model</h3>

# In[226]:


#predictions using the random forest best model :
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_predictions


# In[227]:


#evaluating the random forest best model performance :
best_rf_model.score(X_test, y_test)


# In[228]:


RMSE_best_model = mean_squared_error(y_test, best_rf_predictions, squared = False)
RMSE_best_model


# While evaluating the best random forest model, it appears that the model doesn't generalize to new data. And I decide not to retain the model for predicting churn.

# <h2><b>Conclusion :</b></h2>
# <br>
# The best model fine-tuned for the random forest gave a score of only 0.22, showing that its performance is low. So it was not retained.
# 
# On the contrary, the SVC has been evaluated to provide with 0.795 accuracy score, and it is<b> confirming that is it the best model for the requested business application</b>.
# Its training accuracy (0.79) and test accuracy are similar (0.795) and show that the <b>model generalizes very well to the test set (new data)</b>.
# <br> Besides, its classification reports shows benefits providing a trade-off between the <b>precision of the model (0.63%, the recall 0.48 and optimizing the investments Vs other models</b> in order to retain churning customers.
# 
# But the model has <b>some limitations :</b>
# <br>
# <li>The first one is the size of our Dataset, we <b>have only 7032 rows of customer data</b>, which is a small dataset considering the business application of the predictions, particularly when the <b>most profitable customers segment we need to retain is even the smallest (395</b> i.e 5.6% of our Dataset).</li> 
# <li>Besides, we identified <b>two attributes as important in our model predictions : the tenure and monthly charges</b>. And as we mentioned these are linked to the Customer Lifetime Value (CLV). More Data would be needed to be able to identify the CLV. 
# We thus recommend to hold a <b>second study using unsupervised models to clusters customer segments</b> and link the business retention strategy to  the CLV.</li>
# 
# <br>We can further <b>analyze some attributes combinations or add some features</b> (if we can have more data) to search for a better model performance. </li>
# 
# <br>From business prospective, we recommend the telecom company to run customer surveys and gather customers'feedbacks on services enfavoring churn or enfavoring customers'retention. 
# <br>
# <br>
# Indeed, we have noticed that <b>some services are prone to churn </b>:
# <li>Muliple line subscriptions</li>
# <li>The Internet service with optic Fiber subscription</li>
# <li>Streaming services</li>
# <li>Streaming Movies services</li>
# <br>
# On the contrary, <b>other services, are more enfavoring customer retention or loyalty</b> :
# <br><li>Customers with online security</li>
# <li>Online back-up</li>
# <li>Tech Support. Actually, customers with no Tech Support seems to be enfavoring customer churn.</li>
# <li>The automatic payment methods seems to be a factor that prevent customers from churning (Bank transfer, Credit card, mailed check Vs electronic check (related to Churn?). 
# 
# <br>We would also like to analyze more the data we have, such as <b>modifying the features attributes</b> considering the following adjustments : 
# <li>Customers with Fiber optic or with Multiple Lines Vs "no additional services"
# <li>Group customers with streaming services and customers with streaming movies as customers with streaming Vs "no streaming services".</li>
# <li>Regroup customers with automatic payments methods Vs "not automatic"</li>
# <li>Group customers with full service support : online security & with online Back-up & Tech support Vs customers with "no customer support services".</li>
# <br>
# We need to push further this first shot analysis and modify our features by changing some attributes combinations.
# <br>
# <br>A direction for <b>even further analysis</b> would be to investigate : 
# <li> Who are your loyal customers ? What are their common patterns (payments, services...etc) ?</li> 
# <li>What are the loyal customers most common subscribed services?</li>
# <li>How are your most profitable customers behaving and what are their subscribed services ?</li>
# <li> How to develop a strategy to transform profitable customers into loyal ones ?</li> 
# <li>How can we adjust your services to retain more customers continuously in the customer journey ?</li>

# In[ ]:




