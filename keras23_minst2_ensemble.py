from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(Y_test[0])
print(X_train.shape)    # (60000, 28, 28)
print(X_test.shape)     # (10000, 28, 28)
print(Y_train.shape)
print(Y_test.shape)

from keras.utils import np_utils    # one hot encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255  
#(60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255  

x1_train = X_train[:30000, : , : , :]
x2_train = X_train[30000:, : , : , :]
x1_test = X_test[:5000, : , : , :]
x2_test = X_test[5000:, : , : , :]

# from sklearn.model_selection import train_test_split
# x1_train, x2_train, x1_test, x2_test = train_test_split(
#     X_train, X_test, train_size=0.5, shuffle=False
# )
# y1_train, y2_train, y1_test, y2_test = train_test_split(
#     Y_train, Y_test, train_size=0.5, shuffle = False
# )


# One Hot Encoding : 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 
# 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식 to_categorical()
Y_test = np_utils.to_categorical(Y_test)
Y_train = np_utils.to_categorical(Y_train)  
y1_train = Y_train[:30000,]
y2_train = Y_train[30000:,]
y1_test = Y_test[:5000,]
y2_test = Y_test[5000:,]
# print(Y_train[0])
# print(Y_train.shape)
# print(Y_test.shape)

# 컨볼루션 신경망의 설정
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))   
#  # Conv2D(filter(kernel_size(자르는 단위)), (stride=이동하는 단위))) 
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=2))    # 제일 높은 값만 빼서 다시 만든다. output이 반으로 줄어든다
# model.add(Dropout(0.25))    # summary에는 영향을 미치지 않는다
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))  # 분류 모델을 사용할 땐 마지막에 softmax를 사용해서 나눠준다. 강제로 값을 찾게 하는 함수

input1 = Input(shape=(28, 28, 1))
xx = Conv2D(32, kernel_size=(3,3), activation='relu')(input1)
xx = Conv2D(64, (3, 3), activation='relu')(xx)
xx = MaxPooling2D(pool_size=2)(xx)
xx = Flatten()(xx)
middle1 = Dense(10)(xx)

input2 = Input(shape=(28, 28, 1))
xx = Conv2D(32, kernel_size=(3,3), activation='relu')(input2)
xx = Conv2D(64, (3, 3), activation='relu')(xx)
xx = MaxPooling2D(pool_size=2)(xx)
xx = Flatten()(xx)
middle2 = Dense(10)(xx)

# concatenate  사슬처럼 엮다.
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(5)(merge1)
output1 = Dense(6)(output1)
output1 = Dense(10, activation='softmax')(output1)

output2 = Dense(5)(merge1)
output2 = Dense(8)(output2)
output2 = Dense(10, activation='softmax')(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# categorical_crossentropy : 다중 분류 손실함수 -> softmax를 쓸때는 무조건 loss는 이 함수를 사용해야 한다.

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit([x1_train, x2_train], [y1_train, y2_train], validation_data=([x1_test, x2_test], [y1_test, y2_test]),
                    epochs=2, batch_size=200, verbose=1, # epochs=30
                    callbacks=[early_stopping_callback]) #,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate([x1_test, x2_test], [y1_test, y2_test])[1]))



