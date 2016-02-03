
# coding: utf-8

# In[3]:

#imports
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# In[4]:

train_csv=pd.read_csv('train.csv',header=0)
test_csv=pd.read_csv('test.csv',header=0)


# In[8]:

X_train=train_csv.drop(['target','id'],axis=1)
Y_train=train_csv.target

X_test=test_csv.drop('id',axis=1)


# In[13]:

randomForest=RandomForestClassifier(n_estimators=50,n_jobs=-1)


# In[14]:

randomForest.fit(X_train,Y_train)
predcitons=randomForest.predict(X_test)


# In[16]:

submission=pd.DataFrame({'id':test_csv.id,'target':predcitons})
submission.to_csv('otto.csv',index=False)


# In[27]:

predcitons.size


# In[ ]:




# In[ ]:



