import tensorflow as tf
import numpy as np

# 1. 넘파이 어레이 형식 지정데이터 
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델 준비
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 3 컴파일 훈련하기
# 3-1 손실함수
# 3-2 옵티마이저(로스 최적화) 는 아담이 국룰
# 3-3 
model.compile(loss="mae",optimizer="adam",)
# model.fit 은 훈련시켜라 !! 하는거임 
model.fit(x,y,epochs=1000,steps_per_epoch=100)

# 3 컴파일 훈련하기
# 3-1 손실함수
# 3-2 옵티마이저(로스 최적화) 는 아담이 국룰
# 3-3 
model.compile(loss="mae",optimizer="adam",)
# model.fit 은 훈련시켜라 !! 하는거임 
model.fit(x,y,epochs=1000,steps_per_epoch=100)