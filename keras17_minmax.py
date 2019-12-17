from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000, 400])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x) # evaluate, predict
print(x)

# train과 preduct로 나눌것
# train = 1번째부터 13번째까지
# predict = 14번째
x_train = x[:13]
x_predict = x[13:]
y_train = y[:13]
y_predict = y[13:]
print("x_train.shape : ", x_train.shape)    # (13,3)
print("x_predit.shape : ", x_predict.shape)    # (1,3)

# predit용 데이터
# x_input = array([25,35,45])
# x_input = x_input.reshape(1,3)
# x_input = scaler.transform(x_input) # 앞 전처리에서 fit으로 훈련을 했기때문에 바로 사용가능

# 2. 모델 구성
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape=(3, )))    # activation = 의 defalut값은 linear
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))
# model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
# model.fit(x, y, epochs=180, verbose=2)  # verbose = 0, 1, 2
model.fit(x_train, y_train, epochs=180, verbose=0)  # verbose = 0, 1, 2

# 4. 평가, 예측
# print(x_input.shape)    # (1,3)
# yhat = model.predict(x_input)
print(x_predict)    # (1,3)
yhat = model.predict(x_predict)
print(yhat)
