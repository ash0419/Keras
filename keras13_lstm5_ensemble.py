# keras13_lstm4를 카피해서
# x데이터를 2개로 분리
# 2개의 input 모델인 ensemble을 구현하시오
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)    # 12,3
print("y.shape : ", y.shape)    # 13,

x1 = x[0:3]
x2 = x[10:]
y1 = y[0:3]
y2 = y[10:]

print("x1.shape : ", x1.shape)    # 12,3
print("y1.shape : ", y1.shape)    # 13,
print("x2.shape : ", x2.shape)    # 12,3
print("y2.shape : ", y2.shape)

x1 = x1.reshape(3,3,1)
x2 = x2.reshape(3,3,1)
print(x)
# predit용 데이터
x_input = array([25,35,45])
x_input = x_input.reshape(1,3,1)

# 2. 모델 구성

input1 = Input(shape=(3, 1))
xx = LSTM(40, activation = 'relu')(input1) # 3,1에서 1은 잘라서 작업할 개수
xx = Dense(30)(xx)
xx = Dense(20)(xx)
middle1 = Dense(1)(xx)

input2 = Input(shape=(3, 1))
yy = LSTM(40, activation = 'relu')(input2) # 3,1에서 1은 잘라서 작업할 개수
yy = Dense(30)(yy)
yy = Dense(20)(yy)
middle2 = Dense(1)(yy)
# model.summary()

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(5)(merge1)
output1 = Dense(6)(output1)
output1 = Dense(1)(output1)

output2 = Dense(5)(merge1)
output2 = Dense(8)(output2)
output2 = Dense(1)(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')   # loss값이 patience의 수만큼 나오면 멈춘다 mode='auto'는 loss와 acc를 자동으로 확인
# model.fit(x, y, epochs=180, verbose=0)  # verbose = 0, 1, 2
model.fit([x1, x2], [y1, y2], epochs=5000, verbose=2, callbacks=[early_stopping])  # verbose = 0, 1, 2

yhat1, yhat2 = model.predict([x_input, x_input])
print(yhat1, yhat2)
