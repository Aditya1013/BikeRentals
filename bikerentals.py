
# coding: utf-8

# In[2]:


import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats 
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier 
get_ipython().run_line_magic('matplotlib', 'inline')


os.getcwd()
os.chdir("C:/Users/Adi/Desktop/project2")
os.getcwd()

#get the list of files in the  directy

print(os.listdir(os.getcwd()))

#help('read_csv')

day=pd.read_csv("day.csv")

#Print the `head` of the data
day.head()


# In[3]:


#data insights
day.shape


# In[4]:


#descriptive statistics summary
day['cnt'].describe()


# In[5]:


sns.distplot(day['cnt']);


# In[7]:


#Check whether  variable 'temp'is normal or not
sns.distplot(day['temp']);

#Check whether  variable 'atemp'is normal or not
sns.distplot(day['atemp']);

#Check whether  variable 'hum'is normal or not
sns.distplot(day['hum']);

#Check whether  variable 'windspeed'is normal or not
sns.distplot(day['windspeed']);


#Check whether  variable 'casual'is normal or not
sns.distplot(day['casual']);



#Check whether  variable 'registered'is normal or not
sns.distplot(day['registered']);


# In[7]:


print("Skewness: %f" % day['cnt'].skew())
print("Kurtosis: %f" % day['cnt'].kurt())


# In[8]:


#relation between Numerical Variable 'temp' and target variable 'cnt'

day['temp'].value_counts()

#Now draw scatter plot between 'temp' and 'cnt' variables

var = 'temp'
data = pd.concat([day['cnt'], day[var]], axis=1)
data.plot.scatter(x=var, y='cnt', ylim=(0,9000));


# In[9]:


#relation between Numerical Variable 'atemp' and target variable 'cnt'

day['atemp'].value_counts()

#Now draw scatter plot between 'temp' and 'cnt' variables

var = 'atemp'
data = pd.concat([day['cnt'], day[var]], axis=1)
data.plot.scatter(x=var, y='cnt', ylim=(0,9000));


# In[10]:


#relation between Numerical Variable 'hum' and target variable 'cnt'

day['hum'].value_counts()

#Now draw scatter plot between 'hum' and 'cnt' variables

var = 'hum'
data = pd.concat([day['cnt'], day[var]], axis=1)
data.plot.scatter(x=var, y='cnt', ylim=(0,9000));


# In[11]:


#relation between Numerical Variable 'windspeed' and target variable 'cnt'

day['windspeed'].value_counts()

#Now draw scatter plot between 'windspeed' and 'cnt' variables

var = 'windspeed'
data = pd.concat([day['cnt'], day[var]], axis=1)
data.plot.scatter(x=var, y='cnt', ylim=(0,9000));


# In[12]:


#box plot 'Weekdays' with 'CNT'
var_weekdays = 'weekday'
data = pd.concat([day['cnt'], day[var_weekdays]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var_weekdays, y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);


# In[13]:


#box plot 'weekends' with 'CNT'
var_holiday = 'holiday'
data = pd.concat([day['cnt'], day[var_holiday]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var_holiday, y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);


# In[14]:


#total_missing_values = day.isnull().sum().sort_values(ascending=False)
#total_missing_value

total = day.isnull().sum().sort_values(ascending=False)
percent = (day.isnull().sum()/day.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[15]:


day_1 =  day.copy()


# In[16]:


######################################### Outlier Analysis ##########

day.head()

#plt.boxplot(day_1['casual'])

sns.set(style="whitegrid")
 #tips = sns.load_dataset("tips")
ax = sns.boxplot(x=day['casual'],orient ='h')

#It seems Outliers are present  in  'Casual' variable  but we are keeping as it is , will detect and  conver outliers  during tuning 
#process

# Correlation before  outlier treatment

# Correlation between 'casual' and 'cnt'  before  removal of  outliers
#sns.regplot(x="casual", y="cnt", data=day);

day['casual'].corr(day['cnt'])


# In[17]:


cnames = ['casual']
for i in cnames:
    q75, q25 = np.percentile(day.loc[:,i], [75 ,25])
    iqr = q75 - q25
     
     
        
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
        
print(min)
print(max)

day_out = day.copy()

day_out = day_out.drop(day_out[day_out.loc[:,i] < min].index)
day_out = day_out.drop(day_out[day_out.loc[:,'casual'] > max].index)

# Boxplot for casual after  aoutlier removal

sns.set(style="whitegrid")
 #tips = sns.load_dataset("tips")
ax = sns.boxplot(x=day_out['casual'],orient ='h')

# Correlation between 'casual' and 'cnt'  after  removal of  outliers
sns.regplot(x="casual", y="cnt", data=day_out);

day_out['casual'].corr(day_out['cnt'])


# In[18]:


day.head()
#Selection of numerical feature  based  on pearson corelation 

day_numeric = day.loc[:,['temp','atemp','hum','windspeed','casual','registered','cnt']]
#day_numeric.shape


#draw  correlation matrix between all  numeric variables and analyse  what are the variables are important

day_numeric.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[19]:


sns.set()
cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
sns.pairplot(day_numeric[cols], size = 2.5,kind="reg")
plt.show();

#As per scatter plots and above Correlation  graph there is strong relation 
# Independent variable   'temp' and 'atemp'
# There is a   poor relation between  Independent variable 'hum' and dependent  variable 'cnt'

# so dropping two variables for feature selection

numeric_features = day_numeric.loc[:,['temp', 'windspeed', 'casual', 'registered', 'cnt']]

numeric_features.head()

numeric_features.shape


# In[20]:


# feature  Scaling
#######################################  Normality  Check ######################################

cnames = ['casual','registered']

for i in cnames :
    print(i)
    day[i] = (day[i] - min(day[i]))/(max(day[i]) - min(day[i]))

day.head()


# In[21]:


#deviding  Test and train data  using skilearn   train_test_split 

day_feature_selection = day.drop(['atemp','hum'],axis = 1)
day_feature_selection.shape

from sklearn.model_selection import train_test_split

train, test = train_test_split(day_feature_selection, test_size=0.2)


# In[22]:


from sklearn.tree import DecisionTreeRegressor

train_features_one = train[['season','yr','mnth','holiday','weekday','weathersit','temp','windspeed','casual','registered']].values
train_target_feature = train['cnt'].values
test_feature = test[['season','yr','mnth','holiday','weekday','weathersit','temp','windspeed','casual','registered']].values
test_target_feature= test['cnt'].values
train_features_one
#target_feature

# Implement  decision tree algorithm

# Fit your first decision tree: my_tree_one
my_tree_one = DecisionTreeRegressor()
my_tree_one = my_tree_one.fit(train_features_one, train_target_feature)
print(my_tree_one)



#Decision tree for regression
#fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,2:13], train.iloc[:,13])

#Apply model on test data
predictions_DT = my_tree_one.predict(test_feature)

print(predictions_DT)


# In[23]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

MAPE(test_target_feature, predictions_DT)


# In[24]:


max_depth = 8
min_samples_split =4
my_tree_two = DecisionTreeRegressor(max_depth =max_depth , min_samples_split =min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(train_features_one, train_target_feature)
print(my_tree_two)

predictions_DT_two = my_tree_two.predict(test_feature)

print(predictions_DT_two)

MAPE(test_target_feature,predictions_DT_two)


# In[25]:


rss= ((test_target_feature-predictions_DT_two)**2).sum()
print(rss)

MSE = np.mean((test_target_feature-predictions_DT_two)**2)
print(MSE)

#RMSE
rmse=np.sqrt(MSE)
print(rmse)


# In[26]:


def RMSE(y_test,y_predict):
    mse = np.mean((y_test-y_predict)**2)
    print("Mean Square : ",mse)
    rmse=np.sqrt(mse)
    print("Root Mean Square : ",rmse)
    return rmse

#MAPE 
MAPE(test_target_feature,predictions_DT_two)

#MAPE : 3.87
#RMSE

RMSE(test_target_feature,predictions_DT_two)


# In[27]:


#************************************ Random Forest ************************************************
#here  same features are taking  what we took for the Decision Tree
#train_features_one = train[['season','yr','mnth','holiday','weekday','weathersit','temp','windspeed','casual','registered']].values
#train_target_feature = train['cnt'].values
#test_feature = test[['season','yr','mnth','holiday','weekday','weathersit','temp','windspeed','casual','registered']].values
#test_target_feature= test['cnt'].values
#train_features_one

# Instantiate random forest and train on new features
from sklearn.ensemble import RandomForestRegressor

RF_model_one = RandomForestRegressor(n_estimators= 500, random_state=100).fit(train_features_one,train_target_feature)
#rf_exp.fit(train_features, train_labels)

#print(RF_model)
# Predict the model using predict funtion

RF_predict_one= RF_model_one.predict(test_feature)


# In[28]:


#Evaluate Random forest using  MAPE 

MAPE(test_target_feature,RF_predict_one)


# In[29]:


#Evaluate  Model usinf  RMSE

RMSE(test_target_feature,RF_predict_one)


# In[30]:


import sklearn.feature_selection as fs # feature selection library in scikit-learn


mir_result = fs.mutual_info_regression(train_features_one, train_target_feature) # mutual information


# In[31]:


#tuning  Random FOrest Model

importances = list(RF_model_one.feature_importances_)

print(importances)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(train_features_one, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# In[32]:


train_variables_one_1= train[['season','yr','mnth','holiday','weekday','weathersit','temp','windspeed','casual','registered']]
train_variables_one_1
for name, importance in zip(train_variables_one_1, mir_result):
    print(name, "=", importance)


# In[33]:


# list of x locations for plotting
x_values = list(range(len(mir_result)))

# Make a bar chart
plt.bar(x_values, mir_result, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)

# Tick labels for x axis
plt.xticks(x_values, train_variables_one, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[34]:


train_feature_two = train[["yr" ,"mnth","weekday","workingday","temp","casual","registered"]].values
test_feature_two= test[["yr" ,"mnth","weekday","workingday","temp","casual","registered"]].values
# build random forest model

Rf_model_two = RandomForestRegressor(n_estimators= 500, random_state=100).fit(train_feature_two,train_target_feature)
#rf_exp.fit(train_features, train_labels)

#print(RF_model)
# Predict the model using predict funtion

RF_predict_two= Rf_model_two.predict(test_feature_two)

print(RF_predict_two)


# In[35]:


#Evaluate Random forest using  MAPE 

MAPE(test_target_feature,RF_predict_two)


# In[36]:


#Evaluate  Model usinf  RMSE

RMSE(test_target_feature,RF_predict_two)


# In[37]:


#import  linear regreesion  

import statsmodels.api as sm

#develop Linear Regression model using sm.ols

linear_regression_model = sm.OLS(train_target_feature, train_features_one).fit()

#Summary of model
linear_regression_model.summary()


# In[38]:


MAPE(test_target_feature,predict_LR)
#MAPE  is  0.108

#Predict the model using  RMSE

RMSE(test_target_feature,predict7_LR)

