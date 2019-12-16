from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)    # 13,3
print("y.shape : ", y.shape)    # 13,

x = x.reshape((x.shape[0], x.shape[1], 1))

print(x)
# predit용 데이터
x_input = array([25,35,45])

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1), return_sequences=True)) # return_sequences는 출력 형상은 (none, 3, 노드개수)로 반환
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(30))
model.add(Dense(20))
model.add(Dense(1))
model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='mse', patience=100, mode='auto')   # loss값이 patience의 수만큼 나오면 멈춘다 mode='auto'는 loss와 acc를 자동으로 확인
# model.fit(x, y, epochs=180, verbose=0)  # verbose = 0, 1, 2
model.fit(x, y, epochs=5000, verbose=2, callbacks=[early_stopping])  # verbose = 0, 1, 2


x_input = x_input.reshape(1,3,1)

yhat = model.predict(x_input)
print(yhat)
