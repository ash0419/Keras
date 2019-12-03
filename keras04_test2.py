#   train 데이터와 test데이터를 따로 두는 이유 : 머신이 답만 외우지 않도록 하기 위해서,
#   학습한 자료로는 평가하기가 힘들다
#   train 데이터와 test 데이터는 7:3 비율로 설정
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_predict = np.array([21, 22, 23, 24, 25])

model = Sequential()
model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1)
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_predict)
print(y_predict)