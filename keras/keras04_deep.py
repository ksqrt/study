#!/usr/bin/env python
# coding: utf-8

# In[32]:


# 필수 라이브러리 임포트하기(가독성이 좋음)
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print(tf.__version__)


# In[33]:


# 1. 데이터 준비
x = np.array([1,2,3,4,5]) 
y = np.array([1,2,3,4,5]) 
# shift + del = 라인 삭제


# In[34]:


model = Sequential(
# 인풋 레이어 안에 노드의 계수 레이어의 개수 를 조절 할 수 있습니다.
[   
    Dense(3,input_dim=1),
    Dense(4),
    Dense(5),
    Dense(6),
    Dense(5),
    Dense(4),
    Dense(3),
    Dense(2),
    Dense(1)
]
)



# In[35]:


model.compile(loss="mae",optimizer="adam")
# 3. 모델 학습진행
model.fit(x,y, epochs=100,steps_per_epoch=100)


# In[36]:


print(model.predict([6]))

