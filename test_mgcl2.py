# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:22:03 2022

@author: ma
"""

# Importing the required libraries
import xgboost as xgb
import pandas as pd
import scipy.stats as stats
# First XGBoost model for Pima Indians dataset
import numpy as np
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
# Reading the csv file and putting it into 'df' object
df = pd.read_csv('pore_mgcl.csv')
df.head()

# Putting feature variable to X
X = df.drop('MgCl2',axis=1)
# Putting response variable to y
y = df['MgCl2']
r_score1,r_score2,r_score3=[],[],[]
random2r=[]
rmse_score=[]
rmse_score1,rmse_score2,rmse_score3=[],[],[]
#####################################################################################################
#The training set and testing set were divided into 8:2
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, train_size=0.8, random_state=141)#nacl_117,na2so4_57,71,mgso4_31,mgcl2_141
X_train_all.shape, X_test.shape
#The training set and Validation set were divided into 8:2
X_train, X_ver, y_train, y_ver = train_test_split(X_train_all, y_train_all, train_size=0.8, random_state=45)#nacl_45,na2so4_193,33,mgso4_169,mgcl2_45
X_train.shape, y_ver.shape
#training the model
model = xgb.XGBRegressor(n_estimators=180, 
                         learning_rate=0.05,
                         max_depth=5,
                         silent=True, 
                         objective='reg:squarederror',
                         random_state=7,
                         gamma=0,
                         importance_type='total_gain')  #reg:gamma squarederror
model.fit(X_train, y_train)
#################################################################################################
#用模型测试验证集，这里ver_predict1是验证集的预测结果，ver_error1是验证集的预测值与实际值的差
ver_predict1=model.predict(X_ver)
ver_error1=ver_predict1-y_ver
# Verify the accuracy
from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_ver,ver_predict1))
pearson_r1=stats.pearsonr(y_ver,ver_predict1)
R21=metrics.r2_score(y_ver,ver_predict1)
RMSE1=metrics.mean_squared_error(y_ver,ver_predict1)**0.5

#Draw test plot
font = {"color": "darkred",
        "size": 18,
        "family" : "times new roman"}
font1 = {"color": "black",
        "size": 12,
        "family" : "times new roman"}

Text='r='+str(round(pearson_r1[0],2))
plt.figure(3)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(y_ver,ver_predict1,color='red')
plt.xlabel('True Values',fontdict=font)
plt.ylabel('Predictions',fontdict=font)
Lims=[0,110]
plt.xlim(Lims)
plt.ylim(Lims)
plt.tick_params(labelsize=10)
plt.plot(Lims,Lims,color='black')
plt.grid(False)
plt.title('ion',fontdict=font)
plt.text(2,10,Text,fontdict=font1)
plt.savefig('figure3.png', dpi=100,bbox_inches='tight') 

################################################################################################
# 对测试集进行预测
y_pred = model.predict(X_test)
random_forest_error=y_pred-y_test
# evaluate predictions
from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test,y_pred))
pearson_r=stats.pearsonr(y_test,y_pred)
R2=metrics.r2_score(y_test,y_pred)
RMSE=metrics.mean_squared_error(y_test,y_pred)**0.5
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
print ('r2_score: %.2f' %R2)
rmse_score.append(RMSE)    
#Draw test plot
font = {"color": "darkred",
        "size": 18,
        "family" : "times new roman"}
font1 = {"color": "black",
        "size": 12,
        "family" : "times new roman"}

Text='r='+str(round(pearson_r[0],2))
plt.figure(1)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(y_test,y_pred,color='red')
plt.xlabel('True Values',fontdict=font)
plt.ylabel('Predictions',fontdict=font)
Lims=[0,110]
plt.xlim(Lims)
plt.ylim(Lims)
plt.tick_params(labelsize=10)
plt.plot(Lims,Lims,color='black')
plt.grid(False)
plt.title('ion',fontdict=font)
plt.text(2,10,Text,fontdict=font1)
plt.savefig('figure1.png', dpi=100,bbox_inches='tight')   


plt.figure(2)
plt.clf()
plt.hist(random_forest_error,bins=30)
plt.xlabel('Prediction Error',fontdict=font)
plt.ylabel('Count',fontdict=font)
plt.grid(False)
plt.title('ion',fontdict=font)
plt.savefig('figure2.png', dpi=100,bbox_inches='tight')
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))

# 显示重要特征
plot_importance(model,importance_type=('total_gain'))
pyplot.show()
feature_importance = model.feature_importances_.tolist()
#Calculate the importance of variables

###############################################################################################重复做实验
choice_repeat=[]
for i in range(0,300,1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=i)#89
    X_train.shape, X_test.shape
    
    
# now lets split the data into train and test
#Splitting the data into train and test
#=========================================================================================


    model1 = xgb.XGBRegressor(n_estimators=180, 
                         learning_rate=0.05, 
                         max_depth=5, 
                         silent=True, 
                         objective='reg:squarederror',
                         random_state=7,
                         importance_type='total_gain')
    model1.fit(X_train, y_train)
                 # 对测试集进行预测
    y_pred = model1.predict(X_test)
#################################################################################################
    ver_predict1=model1.predict(X_train)
    ver_error1=ver_predict1-y_train
    # Verify the accuracy
    from sklearn import metrics
    mae1=metrics.mean_absolute_error(y_train,ver_predict1)
    pearson_r1=stats.pearsonr(y_train,ver_predict1)
    R21=metrics.r2_score(y_train,ver_predict1)
    RMSE1=metrics.mean_squared_error(y_train,ver_predict1)**0.5
###########################################################################################################
        ## evaluate predictions
    pearson_r=stats.pearsonr(y_test,y_pred)
    mae=metrics.mean_absolute_error(y_test,y_pred)
    R2=metrics.r2_score(y_test,y_pred)
    RMSE=metrics.mean_squared_error(y_test,y_pred)**0.5
    print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
    print ('r2_score: %.2f' %R2)
    result=[]
    
    result.append(R2)
    result.append(i)
    result.append(mae)
    result.append(RMSE)
    
    feature_importance1 = model1.feature_importances_.tolist()
    for j in feature_importance1:
        result.append(j)
    result.append(R21)
    result.append(mae1)
    result.append(RMSE1)
    choice_repeat.append(result)
    

resule_repeat_choice=sorted(choice_repeat,reverse=True)    
sum_mae,sum_r2,sum_rmse,sum_rd,sum_zeta,sum_bar,sum_con,sum1_mae,sum1_r2,sum1_rmse=[],[],[],[],[],[],[],[],[],[]

for i in resule_repeat_choice[:50]:
    sum_mae.append(i[2])
    sum_r2.append(i[0])
    sum_rmse.append(i[3])
    sum_rd.append(i[4])
    sum_zeta.append(i[5])
    sum_bar.append(i[6])
    sum_con.append(i[7])
    sum1_r2.append(i[8])
    sum1_mae.append(i[9])
    sum1_rmse.append(i[10])
  
all_mae=np.array(sum_mae)
mean_mae=np.mean(all_mae)
std_mae=np.std(all_mae,ddof = 1)

all_r2=np.array(sum_r2)
mean_r2=np.mean(all_r2)
std_r2=np.std(all_r2,ddof = 1)

all_rmse=np.array(sum_rmse)
mean_rmse=np.mean(all_rmse)
std_rmse=np.std(all_rmse,ddof = 1)

all_rd=np.array(sum_rd)
mean_rd=np.mean(all_rd)
std_rd=np.std(all_rd,ddof = 1)

all_zeta=np.array(sum_zeta)
mean_zeta=np.mean(all_zeta)
std_zeta=np.std(all_zeta,ddof = 1)

all_bar=np.array(sum_bar)
mean_bar=np.mean(all_bar)
std_bar=np.std(all_bar,ddof = 1)

all_con=np.array(sum_con)
mean_con=np.mean(all_con)
std_con=np.std(all_con,ddof = 1)

all_mae1=np.array(sum1_mae)
mean_mae1=np.mean(all_mae1)
std_mae1=np.std(all_mae1,ddof = 1)

all_r21=np.array(sum1_r2)
mean_r21=np.mean(all_r21)
std_r21=np.std(all_r21,ddof = 1)

all_rmse1=np.array(sum1_rmse)
mean_rmse1=np.mean(all_rmse1)
std_rmse1=np.std(all_rmse1,ddof = 1)
