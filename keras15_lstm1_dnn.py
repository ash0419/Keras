import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1,11))

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        # aaa.append([item for item in subset])
        temp = []
        for item in subset:
            temp.append(item)
        aaa.append(temp)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("====================")
print(dataset)

x_train = dataset[:,0:-1]
y_train = dataset[:, -1]

print(x_train.shape)    # (6,4)
print(y_train.shape)    # (6, )

# 모델을 마무리해서 완성하시오. fit까지
# 2. 모델구성
model = Sequential()
model.add(Dense(40, activation = 'relu', input_shape=(4,))) # 3,1에서 1은 잘라서 작업할 개수
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=180, batch_size=1)

x2 = np.array([7,8,9,10])

x2 = x2.reshape(1, 4)
print(x2.shape) # (1,4)
y_pred = model.predict(x2)
print(y_pred)

# y_pred을 구하시오.