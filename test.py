#!/usr/bin/python
#coding:utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler # 标准化数据

datas = pd.read_excel('/share/workspaces/zhangyan/work17/20181120.xlsx', sheet_name=1)  #第三批数据未排序1/第三批数据排序2
#datas_test = pd.read_excel('/share/workspaces/zhangyan/work17/20181120.xlsx', sheet_name=2) 


def sort_process(datas):
    r = datas.shape[0]
    for i in range(r):
        datas.iloc[i,0:3] = sorted(datas.iloc[i,0:3])
    for i in range(r):
        datas.iloc[i,3:6] = sorted(datas.iloc[i,3:6])

sort_process(datas)
#print(datas)
#writer = pd.ExcelWriter('/share/workspaces/zhangyan/work17/datas_3_human.xlsx')
#datas.to_excel(writer,'Sheet1',index = False)
#writer.save()

#sort_process(datas_test)
#print(datas_test)
#writer = pd.ExcelWriter('/share/workspaces/zhangyan/work17/datas_3.xlsx')
#datas_test.to_excel(writer,'Sheet1',index = False)
#writer.save()



X = datas.iloc[:,0:-1] 
y = datas.iloc[:,-1] 

#print(X)
print("X shape :",X.shape)
print("y shape :",y.shape)

#过采样（SMOTE）
#from imblearn.over_sampling import SMOTE
#X_resampled_smote, y_resampled_smote = SMOTE(random_state=18).fit_sample(X, y)

print("X_resampled_smote shape :",X_resampled_smote.shape)
print("y_resampled_smote shape :",y_resampled_smote.shape)


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.2, random_state=16) #默认shuffle=True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12) #默认shuffle=True）2（0.2）,  12（0.25）         3(所有不去掉),5,10,16  0（只去掉一个）,3,4,6,,7 ,16，8（所有去掉）
print(X_test)



#X_test = datas_test.iloc[:,0:-1].values 
#y_test = datas_test.iloc[:,-1].values  

print("X_train shape :",X_train.shape)
print("X_test shape :",X_test.shape)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

LR = LogisticRegression()  
param_test= {'C':[1e-11,1e-9,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.25,0.3,0.5,0.6,0.7,0.8,0.9,1,1.3,1.5,1.6,1.7,1.8,1.9,2,3,3.7,4,4.3,4.5,4.7,5,10,100,1000,1e4,1e5,1e6,1e7],'penalty': ['l1','l2']}
#'class_weight':['balanced'],


from sklearn.model_selection import GridSearchCV
gcv = GridSearchCV(estimator = LR , param_grid = param_test, scoring='recall', cv=5) #  roc_auc  recall

gcv.fit(X_train,y_train)
print(gcv.best_params_, gcv.best_score_)
y_predict = gcv.predict(X_test)

print('y_test:')
print(y_test)
print('y_predict:', pd.DataFrame(y_predict))


print(gcv.predict_proba(X_test)) 
print('Accuracy of gcv_train :', gcv.score(X_train, y_train))
print('Accuracy of gcv_test  :', gcv.score(X_test, y_test))
print('accuracy_score       :', accuracy_score(y_test, y_predict))
print('precision_score      :', precision_score(y_test, y_predict))
print('recall_score         :', recall_score(y_test, y_predict))
print('f1_score             :', f1_score(y_test, y_predict))
print('confusion_matrix:')
print(confusion_matrix(y_test, y_predict))  #横向标签，纵向预测
print(confusion_matrix(y_test, y_predict).T) #横向预测，纵向标签
print(classification_report(y_test, y_predict, target_names=['label_0', 'label_1']))
print(gcv)

