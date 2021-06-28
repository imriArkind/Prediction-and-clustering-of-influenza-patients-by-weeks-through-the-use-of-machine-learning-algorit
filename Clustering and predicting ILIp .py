#!/usr/bin/env python
# coding: utf-8

# # TASK D
# * Idan Maoz 
# * Imri Arkind 

# # influenza-like illness dataset
# **In this notebook we predictthe ILIp 4 weeks ahead**
# 
# **we have 1235 Observations and 10 Predictive variables:**
# 1. YEAR , the year between 1997-2001
# 2. WEEK, the week between 1 and 53
# 3. Age 0-4, the patient age is between 0-4
# 4. Age 25-49, the patient age is between 25-49
# 5. Age 25-64, the patient age is between 25-64
# 6. Age 5-24, the patient age is between 5-24
# 7. Age 50-64, the patient age is between 50-64
# 8. Age 65, the patient age is over 65
# 9. ILITOTAL, total patient that have influenza-like illness
# 10. TOTAL PATIENTS, total of the patients

# ## Imports

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import math
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture


# In[2]:


seed=123
ili = pd.read_csv("ILINET.csv") 
count=0
season=[]
for row_index, row in ili.iterrows():
    if(row['WEEK']==40):
        count=count+1 
    season.append(count)
ili['season']=season

ili


# drop the columns and rows that are not necessary 

# In[3]:


ili=ili.drop(['REGION TYPE','REGION','% WEIGHTED ILI','%UNWEIGHTED ILI','NUM. OF PROVIDERS'],axis=1)
toSave=[]
for i in range(0,1234):
    if ili["AGE 0-4"][i]==0:
        toSave.append(i)
ili=ili.drop(toSave)

ili=ili.reset_index()
ili['ILIp']=(ili['ILITOTAL']/ili['TOTAL PATIENTS'])*100

ili


# ### Handling Missing Values

# In[4]:


x_train1=ili.loc[532:1139][['AGE 0-4','AGE 5-24','AGE 65']]
y_train1=ili.loc[532:1139]['AGE 50-64']
x_test1=ili.loc[0:531][['AGE 0-4','AGE 5-24','AGE 65']]

x_train2=ili.loc[532:1139][['AGE 0-4','AGE 5-24','AGE 65']]
y_train2=ili.loc[532:1139]['AGE 25-49']
x_test2=ili.loc[0:531][['AGE 0-4','AGE 5-24','AGE 65']]

x_train3=ili.loc[0:531][['AGE 0-4','AGE 5-24','AGE 65']]
y_train3=ili.loc[0:531]['AGE 25-64']
x_test3=ili.loc[532:1139][['AGE 0-4','AGE 5-24','AGE 65']]


# we decided to choose to make SVR regression to handeling the missing values

# In[5]:


regr1 = make_pipeline( SVR(C=1.0, epsilon=0.2))
regr1.fit(x_train1, y_train1)
y_pred1=regr1.predict(x_test1)

regr2 = make_pipeline( SVR(C=1.0, epsilon=0.2))
regr2.fit(x_train2, y_train2)
y_pred2=regr2.predict(x_test2)

regr3 = make_pipeline( SVR(C=1.0, epsilon=0.2))
regr3.fit(x_train3, y_train3)
y_pred3=regr3.predict(x_test3)


# In[6]:


ili['AGE 50-64'][0:532].replace(['X'],[y_pred1], inplace=True)

ili['AGE 25-49'][0:532].replace(['X'],[y_pred2], inplace=True)

ili['AGE 25-64'][532:1140].replace(['X'],[y_pred3], inplace=True)

ili['AGE 0-4'] = pd.to_numeric(ili['AGE 0-4'], downcast="float")
ili['AGE 25-49'] = pd.to_numeric(ili['AGE 25-49'], downcast="float")
ili['AGE 25-64'] = pd.to_numeric(ili['AGE 25-64'], downcast="float")
ili['AGE 5-24'] = pd.to_numeric(ili['AGE 5-24'], downcast="float")
ili['AGE 50-64'] = pd.to_numeric(ili['AGE 50-64'], downcast="float")
ili['AGE 65'] = pd.to_numeric(ili['AGE 65'], downcast="float")

x_train=ili.drop(['ILITOTAL','TOTAL PATIENTS','ILIp'],axis=1)


ili


# In[7]:


count=0
season=[]
for row_index, row in ili.iterrows():
    if(row['WEEK']==40):
        count=count+1 
    season.append(count)
ili['season']=season       
ili


# In[8]:


seasonGroup=ili.groupby('season')
season_df=pd.DataFrame().T
for name, group in seasonGroup:
    new_row = pd.DataFrame([group.mean()])
    season_df = season_df.append(new_row, ignore_index=False)
season_df=season_df.drop(['index','YEAR','WEEK','ILITOTAL','TOTAL PATIENTS'],axis=1)
season_df.index=season_df['season']


# In[9]:



season_df_scaled = pd.DataFrame(scale(season_df), index=season_df.index, columns=season_df.columns)
season_df_scaled['ILIp']=season_df['ILIp']
season_df_scaled 


# ## Q1

# ### Clustering

# #### kmeans

# In[10]:


wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=seed)
    kmeans.fit(season_df_scaled )
    wcss.append(kmeans.inertia_)
plt.plot(range(2, 10), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[11]:


kmeans = KMeans(n_clusters=5, random_state=seed)
clusters = kmeans.fit_predict(season_df_scaled)
season_df_scaled['clusters']=clusters
season_df_scaled


# # Clustering for each season, its easy to see the separation between the seasons

# In[12]:


groups2=season_df_scaled.groupby('clusters')
for name, group in groups2:
    plt.plot(group.index, group['ILIp'], marker='o', linestyle='', label=name)
    
plt.title('Kmeans clustering')
plt.xlabel('season')
plt.ylabel('ILIp')
plt.show()


# #### hierarchical clustering

# In[13]:


dendrogram=sch.dendrogram(sch.linkage(season_df_scaled,method='complete'))
plt.title('Dendrogram')
plt.xlabel('values')
plt.ylabel('Euclidean distances')
plt.show


# In[14]:


hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
y_hc=hc.fit_predict(season_df_scaled)

plt.scatter(season_df_scaled.loc[y_hc==0]['season'], season_df_scaled.loc[y_hc==0]['ILIp'], s=100, c='red', label ='Cluster 1')
plt.scatter(season_df_scaled.loc[y_hc==1]['season'], season_df_scaled.loc[y_hc==1]['ILIp'], s=100, c='blue', label ='Cluster 2')
plt.scatter(season_df_scaled.loc[y_hc==2]['season'], season_df_scaled.loc[y_hc==2]['ILIp'], s=100, c='green', label ='Cluster 3')
plt.scatter(season_df_scaled.loc[y_hc==3]['season'], season_df_scaled.loc[y_hc==3]['ILIp'], s=100, c='yellow', label ='Cluster 1')
plt.scatter(season_df_scaled.loc[y_hc==4]['season'], season_df_scaled.loc[y_hc==4]['ILIp'], s=100, c='black', label ='Cluster 2')


plt.title('Clusters of Values (Hierarchical Clustering Model-complete)')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.show()


# #### GMM

# In[15]:


gmm = GaussianMixture(n_components=5).fit(season_df_scaled)
labels = gmm.predict(season_df_scaled)
plt.scatter(season_df_scaled.index, season_df_scaled['ILIp'], c=labels, s=40, cmap='viridis')

plt.title('GMM clustering')
plt.xlabel('season')
plt.ylabel('ILIp')
plt.show()


# # Q1 Summary: all the alghoritems gave us same results. there are 4 main clusters, and the flu seems more alike in  subsequent years

# ## Q2

# create train and test

# In[16]:


train=ili[ili['index']<940]

test=ili[ili['index']>=940]


# To each observation we will add the value that belongs to the observation in 4 weeks

# In[17]:


weeks4_ahead=[]
weeks3_ahead=[]
weeks2_ahead=[]
weeks1_ahead=[]
for i in range(train.shape[0]-4):
    weeks4_ahead.append(train['ILIp'][i+4])
for i in range(train.shape[0]-3):
    weeks3_ahead.append(train['ILIp'][i+3])
for i in range(train.shape[0]-2):
    weeks2_ahead.append(train['ILIp'][i+2])
for i in range(train.shape[0]-1):
    weeks1_ahead.append(train['ILIp'][i+1])

weeks4_ahead.append(train['ILIp'][841])   
weeks4_ahead.append(train['ILIp'][842]) 
weeks4_ahead.append(train['ILIp'][843]) 
weeks4_ahead.append(train['ILIp'][844])
weeks3_ahead.append(train['ILIp'][842]) 
weeks3_ahead.append(train['ILIp'][843]) 
weeks3_ahead.append(train['ILIp'][844])
weeks2_ahead.append(train['ILIp'][843]) 
weeks2_ahead.append(train['ILIp'][844])
weeks1_ahead.append(train['ILIp'][844])


# In[18]:


train=train.copy()
train['4 weeks ahead']=np.array(weeks4_ahead)
train['3 weeks ahead']=np.array(weeks3_ahead)
train['2 weeks ahead']=np.array(weeks2_ahead)
train['1 weeks ahead']=np.array(weeks1_ahead)
train


# In[19]:


test


# In[20]:



weeks3__ahead=[]
weeks2__ahead=[]
weeks1__ahead=[]
y_test=[]
for i in range(845,1136):
    weeks3__ahead.append(test['ILIp'][i+3])
    weeks2__ahead.append(test['ILIp'][i+2])
    weeks1__ahead.append(test['ILIp'][i+1])
    y_test.append(test['ILIp'][i])
y_test.append(test['ILIp'][1136])
y_test.append(test['ILIp'][1137])
y_test.append(test['ILIp'][1138])
y_test.append(test['ILIp'][1139])

y_test=np.array(y_test)
x_test = pd.DataFrame({'3 weeks ahead': weeks3__ahead, '2 weeks ahead': weeks2__ahead,'1 weeks ahead': weeks1__ahead})


# In[21]:


x_train=train[['3 weeks ahead','2 weeks ahead','1 weeks ahead']]
y_train=train['4 weeks ahead']


# ### HR function

# In[22]:


def calculate_HR(y_real,y_pred):
    real_hr=[]
    pred_hr=[]
    for i in range(1,len(y_real)):
        real_hr.append(y_real[i]-y_real[i-1])
        pred_hr.append(y_pred[i]-y_pred[i-1]) 
    count=0
    for i in range(len(real_hr)):
        if(np.sign(real_hr[i])==np.sign(pred_hr[i])):
            count=count+1
            
    return (count/(len(y_real)-1))*100
        
    


# ### Baseline Predictor

# In[23]:


base_dic=dict()
predILIpgroup=[]
for row_index, row in test.iterrows():
    base_dic.update({row['WEEK']: [0,0]})
    
for row_index, row in test.iterrows():
    base_dic.update({row['WEEK']: [(row['ILIp']+base_dic[row['WEEK']][0]),base_dic[row['WEEK']][1]+1]})
    predILIpgroup.append(base_dic[row['WEEK']][0]/base_dic[row['WEEK']][1])

r_bp=np.corrcoef(y_test,predILIpgroup)[0][1]
RMSE_bp=math.sqrt(mean_squared_error(y_test,predILIpgroup))
RMSPE_bp=np.sqrt(np.mean(np.square(((y_test - predILIpgroup) / y_test))))*100
MAPE_bp=np.mean(np.abs((y_test - predILIpgroup) / y_test)) * 100
HR_bp=calculate_HR(y_test,predILIpgroup)   
       


# ### SVR 

# In[24]:


regSVR = make_pipeline(SVR(C=1.0, epsilon=0.2))
fitSVR=regSVR.fit(x_train,y_train)
y_predSVR=regSVR.predict(x_test)

r_svr=np.corrcoef(y_test[4:295],y_predSVR[0:291])[0][1]
RMSE_svr=math.sqrt(mean_squared_error(y_test[4:295],y_predSVR[0:291]))
RMSPE_svr=np.sqrt(np.mean(np.square(((y_test[4:295]-y_predSVR[0:291]) / y_test[4:295])), axis=0))*100
MAPE_svr=np.mean(np.abs((y_test[4:295]-y_predSVR[0:291]) /y_test[4:295])) * 100
HR_svr=calculate_HR(y_test[4:295],y_predSVR[0:291])


# ### KNN using Cross Validation

# In[25]:


params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23,25,27]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
print(model.best_params_)
modelKNN = neighbors.KNeighborsRegressor(n_neighbors = 23)
modelFITKNN=modelKNN.fit(x_train,y_train)
y_predKNN=modelFITKNN.predict(x_test)

r_knn=np.corrcoef(y_test[4:295],y_predKNN[0:291])[0][1]
RMSE_knn=math.sqrt(mean_squared_error(y_test[4:295],y_predKNN[0:291]))
RMSPE_knn=np.sqrt(np.mean(np.square(((y_test[4:295]-y_predKNN[0:291]) / y_test[4:295])), axis=0))*100
MAPE_knn=np.mean(np.abs((y_test[4:295]-y_predKNN[0:291]) /y_test[4:295])) * 100
HR_knn=calculate_HR(y_test[4:295],y_predKNN[0:291])


# ### Linear Regression WITH time series (we did it for the 4 last IPIs)

# In[26]:


model = LinearRegression().fit(x_train, y_train) 
y_predLR=model.predict(x_test)

r_lr=np.corrcoef(y_test[4:295],y_predLR[0:291])[0][1]
RMSE_lr=math.sqrt(mean_squared_error(y_test[4:295],y_predLR[0:291]))
RMSPE_lr=np.sqrt(np.mean(np.square(((y_test[4:295]-y_predLR[0:291]) / y_test[4:295])), axis=0))*100
MAPE_lr=np.mean(np.abs((y_test[4:295]-y_predLR[0:291]) /y_test[4:295])) * 100
HR_lr=calculate_HR(y_test[4:295],y_predLR[0:291])


# ### Comparing 

# In[28]:


result_df=pd.DataFrame([r_bp,RMSE_bp,RMSPE_bp,MAPE_bp,HR_bp]).T
result_df=result_df.rename(columns = {0:'r',1:'RMSE',2:'RMSPE',3:'MAPE',4:'HR'},index={0: 'Baseline Predictor'} ,inplace = False)

new_row = pd.Series(data={'r':r_svr, 'RMSE':RMSE_svr, 'RMSPE':RMSPE_svr,'MAPE':MAPE_svr,'HR':HR_svr},name='SVR')
result_df = result_df.append(new_row, ignore_index=False)

new_row = pd.Series(data={'r':r_knn, 'RMSE':RMSE_knn, 'RMSPE':RMSPE_knn,'MAPE':MAPE_knn,'HR':HR_knn},name='KNN')
result_df = result_df.append(new_row, ignore_index=False)

new_row = pd.Series(data={'r':r_lr, 'RMSE':RMSE_lr, 'RMSPE':RMSPE_lr,'MAPE':MAPE_lr,'HR':HR_lr},name='Linear Regression')
result_df = result_df.append(new_row, ignore_index=False)


result_df=result_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
result_df


# ### Conclusion

# we can see from the table that the best algorithm with the less RMSE are Linear Regression
