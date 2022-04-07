#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns


# In[32]:


df=pd.read_csv(r"C:\Users\Richa\Desktop\fake-news\train.csv")
df


# In[33]:


df.isnull().sum()


# In[34]:


df.shape


# In[35]:


df.info()


# In[36]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="rainbow")


# In[37]:


df=df.dropna()


# In[38]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="rainbow")


# In[39]:


X=df.drop("label",axis=1)
X.head()


# In[40]:


Y=df.iloc[:,-1:]
Y.head()


# In[41]:


from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM
from keras.layers import Dense


# In[42]:


voc_size=5000


# In[43]:


msg=X.copy()


# In[44]:


msg.reset_index(inplace=True)


# In[45]:


msg.head(10)


# In[46]:


import re
from nltk.corpus import stopwords


# # Data Preprocessing

# In[47]:


from nltk.stem.porter import PorterStemmer


# In[48]:


ps=PorterStemmer()


# In[49]:


corpus=[]
for i in range(0,len(msg)):
    x=re.sub("[^a-zA-Z]"," ",msg["title"][i])
    x=x.lower()
    x=x.split()
    x=[ps.stem(word) for word in x if not word in stopwords.words("english")]
    x=" ".join(x)
    corpus.append(x)


# In[50]:


nltk.download('stopwords')


# In[51]:


corpus[1]


# In[52]:


encoding=[one_hot(words,voc_size) for words in corpus]
encoding


# In[53]:


lenght=20
emb=pad_sequences(encoding,padding="pre",maxlen=lenght)
print(emb)


# In[54]:


emb[1]


# In[56]:


emb_feature=40
model=Sequential()
model.add(Embedding(voc_size,emb_feature,input_length=lenght))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))


# In[59]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[60]:


model.summary()


# In[61]:


X_result=np.array(emb)
Y_result=np.array(Y)
print(X_result.shape,Y_result.shape)


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train,X_test,Y_train,Y_test=train_test_split(X_result,Y_result,test_size=0.33,random_state=42)


# # Train

# In[64]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=125,batch_size=32)


# In[65]:


from keras.layers import Dropout


# In[66]:


emb_feature=40
model=Sequential()
model.add(Embedding(voc_size,emb_feature,input_length=lenght))
model.add(LSTM(100))
model.add(Dropout(0.30))
model.add(Dense(1,activation='sigmoid'))


# In[67]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[68]:


X_result=np.array(emb)
Y_result=np.array(Y)
print(X_result.shape,Y_result.shape)


# In[69]:


X_train,X_test,Y_train,Y_test=train_test_split(X_result,Y_result,test_size=0.33,random_state=42)


# In[70]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=125,batch_size=32)

