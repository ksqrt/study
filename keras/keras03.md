```python
# 0. 텐서플로우 넘파이 임포트
import numpy as np
import tensorflow as tf

print(tf.__version__)
```

    2.7.4

```python
# 1. 데이터 준비
x = np.array([i for i in range(1,6)])
y = np.array([1,2,3,5,4])
print(x)
print(y)
```

    [1 2 3 4 5]
    [1 2 3 4 5]

```python
# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 모델은 순차적이며 input 노드 1개 output 노드 1개 로 구성
# 손실함수 mae, 옵티마이저는 아담이 국룰
model = Sequential()
model.add(Dense(1,input_dim=1))
model.compile(loss="mae",optimizer="adam")
```

```python
# 3. 모델 학습진행
model.fit(x,y, epochs=100,steps_per_epochs=100)
result = model.predict([6])

print("6의 예측값 :", result)
```

    Epoch 1/100
    100/100 [==============================] - 0s 574us/step - loss: 0.1886
    Epoch 2/100
    100/100 [==============================] - 0s 554us/step - loss: 0.0312
    Epoch 3/100
    100/100 [==============================] - 0s 897us/step - loss: 0.0203
    Epoch 4/100
    100/100 [==============================] - 0s 463us/step - loss: 0.0101
    Epoch 5/100
    100/100 [==============================] - 0s 473us/step - loss: 0.0024
    Epoch 6/100
    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 10000 batches). You may need to use the repeat() function when building your dataset.
    100/100 [==============================] - 0s 160us/step - loss: 0.0024
    6의 예측값 : [[5.999466]]
