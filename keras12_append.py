# 함수형 모델로 사용
# 1.데이터
import numpy as np
x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711,811), range(100)])

x2 = np.array([range(100, 200), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)   # x1 = x1.T
x2 = np.transpose(x2)   # x2 = x2.T
y1 = np.transpose(y1)   # y1 = y1.T
y2 = np.transpose(y2)   # y2 = y2.T

print(x1.shape) # (100, 3)
print(x2.shape) # (100, 3)
print(y1.shape) # (100, 3)
print(y2.shape) # (100, 3)

# 자료를 하나로 합치기
x = np.hstack([x1, x2])  # np.concatenate([x1, x2])
print(x.shape)  # (100, 6)
y = np.hstack([y1, y2])  # np.concatenate([y1, y2])
print(y.shape)  # (100, 6)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, train_size=0.6, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle = False
)

print(x_test.shape)    #(20, 6)

# 2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input = Input(shape=(6, ))
xx = Dense(250, activation='relu')(input)
xx = Dense(200)(xx)
xx = Dense(140)(xx)
xx = Dense(200)(xx)
xx = Dense(90)(xx)
xx = Dense(110)(xx)
xx = Dense(100)(xx)
output = Dense(6)(xx)

model = Model(inputs = input, outputs = output)
model.summary()

# 3.훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val,y_val))

# 4.평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
print("loss : ", loss)

print(x_test)
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
