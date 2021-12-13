#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[40]:


Stroke = pd.read_csv("C:/Users/ASUS/Tugas Mandiri SPADA DIKTI/Stroke/healthcare-dataset-stroke-data.csv")
Stroke.head()


# In[41]:


Stroke.info()


# In[42]:


#Data Preparation


# In[43]:


Stroke.isnull().sum()


# In[44]:


Stroke.dropna(inplace=True)


# In[45]:


Stroke.isnull().sum()


# In[46]:


#Data Cleaning


# In[47]:


Stroke = Stroke.drop('id',axis=1)
Stroke.info()


# In[48]:


Stroke['Residence_type'].value_counts()


# In[49]:


Stroke['ever_married'] = Stroke['ever_married'].map({'Yes':1,'No':0})
Stroke['Residence_type'] = Stroke['Residence_type'].map({'Urban':1,'Rural':0})
Stroke.info()


# In[50]:


temp = pd.get_dummies(Stroke['gender'])
Stroke = pd.concat([Stroke,temp],axis=1)
Stroke.drop('gender',axis=1,inplace=True)

temp = pd.get_dummies(Stroke['work_type'])
Stroke = pd.concat([Stroke,temp],axis=1)
Stroke.drop('work_type',axis=1,inplace=True)

temp = pd.get_dummies(Stroke['smoking_status'])
Stroke = pd.concat([Stroke,temp],axis=1)
Stroke.drop('smoking_status',axis=1,inplace=True)


# In[51]:


#Not deleting the first column of dummy variables as I intend to perform EDA on them. 


# In[52]:


Stroke.info()


# In[53]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[54]:


imputer = IterativeImputer()
col = Stroke.columns
Stroke = imputer.fit_transform(Stroke)
Stroke = pd.DataFrame(Stroke,columns=col)


# In[55]:


Stroke.head()


# In[56]:


Stroke.info()


# In[57]:


#EDA


# In[58]:


plt.figure(figsize=(20,15))
sns.heatmap(Stroke.corr(),annot=True)
plt.show()


# In[59]:


#Stroke has a relatively high correlation compared to the other variables with Age, hypertension, heart disease, glucose level 
#and marrital status. 


# In[60]:


plt.figure(figsize=(10,5))
sns.barplot(x='stroke',y='age',data=Stroke)
plt.show()


# In[61]:


plt.figure(figsize=(10,5))
sns.barplot(x='stroke',y='avg_glucose_level',data=Stroke)
plt.show()


# In[62]:


plt.figure(figsize=(10,5))
sns.barplot(x='stroke',y='bmi',data=Stroke)
plt.show()


# In[63]:


plt.figure(figsize=(10,5))
sns.countplot(x='stroke',hue='heart_disease',data=Stroke)
plt.show()


# In[64]:


plt.figure(figsize=(10,5))
sns.countplot(x='stroke',hue='ever_married',data=Stroke)
plt.show()


# In[65]:


plt.figure(figsize=(10,5))
sns.countplot(x='stroke',hue='children',data=Stroke)
plt.show()


# In[66]:


#Strokes occur with higher age and glucose level as well as bmi to a much lower degree. 


# In[67]:


plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.barplot(x='stroke',y='Private',data=Stroke)
plt.subplot(2,2,2)
sns.barplot(x='stroke',y='Self-employed',data=Stroke)
plt.subplot(2,2,3)
sns.barplot(x='stroke',y='Govt_job',data=Stroke)
plt.subplot(2,2,4)
sns.barplot(x='stroke',y='Never_worked',data=Stroke)
plt.show()


# In[68]:


#Greater occurance of stroke when self emplyed or private. 


# In[69]:


plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.barplot(x='stroke',y='smokes',data=Stroke)
plt.subplot(2,2,2)
sns.barplot(x='stroke',y='formerly smoked',data=Stroke)
plt.subplot(2,2,3)
sns.barplot(x='stroke',y='never smoked',data=Stroke)
plt.subplot(2,2,4)
sns.barplot(x='stroke',y='Unknown',data=Stroke)
plt.show()


# In[70]:


#greater occurance of stroke when smokes and formerly smoked. However between the two greater occurance when the person has 
#formery smoked


# In[71]:


Stroke['stroke'].value_counts()


# In[72]:


#Prediction


# In[73]:


X = Stroke.drop('stroke',axis=1)
y = Stroke['stroke']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100)
X_train.head()


# In[74]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV


# In[ ]:


Model = RandomForestClassifier(random_state=100,n_jobs=-1,class_weight='balanced')

params = {'n_estimators':[50,100,200],
          'min_samples_leaf':[10,30,50,70,100,200],
          'max_depth':[3,5,10,20,40,60,100],
          'max_features':[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7],
          'criterion':["gini","entropy"]}

grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='recall')
grid_search.fit(X_train,y_train)


# In[ ]:


Model_best = grid_search.best_estimator_


# In[ ]:


from sklearn.metrics import f1_score,recall_score,plot_roc_curve


# In[ ]:


plot_roc_curve(Model_best,X_train,y_train)
y_train_pred = Model_best.predict(X_train)

print(f1_score(y_train,y_train_pred))
print(recall_score(y_train,y_train_pred))


# In[ ]:


plot_roc_curve(Model_best,X_test,y_test)
y_test_pred = Model_best.predict(X_test)

print(f1_score(y_test,y_test_pred))
print(recall_score(y_test,y_test_pred))


# In[ ]:


Model = RandomForestClassifier(random_state=100,n_jobs=-1,class_weight='balanced')

params = {'n_estimators':[50,100,200],
          'min_samples_leaf':[10,30,50,70,100,200],
          'max_depth':[3,5,10,20,40,60,100],
          'max_features':[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7],
          'criterion':["gini","entropy"]}

grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='f1')
grid_search.fit(X_train,y_train)


# In[ ]:


Model_best = grid_search.best_estimator_


# In[ ]:


plot_roc_curve(Model_best,X_train,y_train)
y_train_pred = Model_best.predict(X_train)

print(f1_score(y_train,y_train_pred))
print(recall_score(y_train,y_train_pred))


# In[ ]:


plot_roc_curve(Model_best,X_test,y_test)
y_test_pred = Model_best.predict(X_test)

print(f1_score(y_test,y_test_pred))
print(recall_score(y_test,y_test_pred))


# In[ ]:


Feature_importance = pd.DataFrame({'Feature':X_train.columns,'Importance':Model_best.feature_importances_})
Feature_importance = Feature_importance.sort_values(by='Importance',ascending=False)
Feature_importance = Feature_importance.set_index('Feature')


# In[ ]:


Feature_importance


# In[ ]:


#Using recall as the metric for validation as well as accuracy. 
#However giving a lot more importance to recall as predicting True positives are the most important.


# In[ ]:


col = X.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[col])
X_scaled = pd.DataFrame(X_scaled,columns=col)
X_scaled.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,train_size=0.7,random_state=100)
X_train.head()


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


Model = SVC(random_state=100,class_weight='balanced')

params = {'C':[0.01,0.1,1,10,100,1000,10000],
          'gamma':[10,1,0.1,0.001,0.0001,0.00001]}

grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='recall')
grid_search.fit(X_train,y_train)


# In[ ]:


Model_best = grid_search.best_estimator_


# In[ ]:


plot_roc_curve(Model_best,X_train,y_train)
y_train_pred = Model_best.predict(X_train)

print(f1_score(y_train,y_train_pred))
print(recall_score(y_train,y_train_pred))


# In[ ]:


plot_roc_curve(Model_best,X_test,y_test)
y_test_pred = Model_best.predict(X_test)

print(f1_score(y_test,y_test_pred))
print(recall_score(y_test,y_test_pred))


# In[ ]:


Model = SVC(random_state=100,class_weight='balanced')

params = {'C':[0.01,0.1,1,10,100,1000,10000],
          'gamma':[10,1,0.1,0.001,0.0001,0.00001]}

grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='f1')
grid_search.fit(X_train,y_train)


# In[ ]:


Model_best = grid_search.best_estimator_


# In[ ]:


plot_roc_curve(Model_best,X_train,y_train)
y_train_pred = Model_best.predict(X_train)

print(f1_score(y_train,y_train_pred))
print(recall_score(y_train,y_train_pred))


# In[ ]:


plot_roc_curve(Model_best,X_test,y_test)
y_test_pred = Model_best.predict(X_test)

print(f1_score(y_test,y_test_pred))
print(recall_score(y_test,y_test_pred))


# In[ ]:




