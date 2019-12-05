# Multi Layer Perceptron
# 1.데이터
import numpy as np

# x = np.array(range(1,101))
# y = np.array(range(1,101))
x = np.array([range(1,101), range(101,201)])
y = np.array([range(201,301)])
print(x.shape)  # (2, 100)

x = np.transpose(x)
y = np.transpose(y) # (1, 100) -> (100, 1)
print(x.shape)  # (100, 2)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split    # 컬럼이 다르다고 해도 똑같은 비율로 나눠줌
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle = False
)   # train_size가 먼저 적용


# 2.모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(40, input_shape=(2, ), activation='relu'))  # (?,2)
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

# 3.훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# 4.평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
print("loss : ", loss)

# aaa = np.array([[101,102,103], [201,202,203]])
# print(aaa)
# aaa = aaa.transpose()
# print(aaa)

y_predict = model.predict(x_test)
print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ",r2_y_predict)
