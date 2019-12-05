# 함수형 모델로 사용
# 1.데이터
import numpy as np
x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711,811), range(100)])

x2 = np.array([range(100, 200), range(311,411), range(100,200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

print(x1.shape) # (100, 3)
print(x2.shape) # (100, 3)
print(y1.shape) # (100, 3)
print(y2.shape) # (100, 3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, train_size=0.6, shuffle=False
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state=66, test_size=0.5, shuffle = False
)   

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=66, train_size=0.6, shuffle=False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
    x2_test, y2_test, random_state=66, test_size=0.5, shuffle = False
)

print(x2_test.shape)    #(20, 3)

# 2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(3, ))
dense1 = Dense(40, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(25)(dense2)
dense4 = Dense(5)(dense3)
middle1 = Dense(3)(dense4)

input2 = Input(shape=(3,))
xx = Dense(40, activation='relu')(input1)
xx = Dense(30)(xx)
xx = Dense(25)(xx)
xx = Dense(5)(xx)
middle2 = Dense(3)(xx)

# concatenate  사슬처럼 엮다.
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()
'''
# 3.훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

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
'''