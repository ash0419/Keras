from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(Y_test[0])
print(X_train.shape)    # (60000, 28, 28)
print(X_test.shape)     # (10000, 28, 28)
print(Y_train.shape)
print(Y_test.shape)

from keras.utils import np_utils    # one hot encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255  
#(60000, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255  
# /255는 minmax로 데이터를 전처리 한 것
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255

# One Hot Encoding : 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 
# 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식 to_categorical()
Y_train = np_utils.to_categorical(Y_train)  # 분류 할 때 강제성을 준다.
Y_test = np_utils.to_categorical(Y_test)    
print(Y_train[0])
print(Y_train.shape)
print(Y_test.shape)

# 컨볼루션 신경망의 설정
model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
model.add(Dense(32, input_shape=(28 * 28, ), activation='relu'))   
# Conv2D(filter(kernel_size(자르는 단위)), (stride=이동하는 단위))) 
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))    # 제일 높은 값만 빼서 다시 만든다. output이 반으로 줄어든다
# model.add(Dropout(0.25))    # summary에는 영향을 미치지 않는다
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 분류 모델을 사용할 땐 마지막에 softmax를 사용해서 나눠준다. 강제로 값을 찾게 하는 함수

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# categorical_crossentropy : 다중 분류 손실함수 -> softmax를 쓸때는 무조건 loss는 이 함수를 사용해야 한다.

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=100, batch_size=200, verbose=1, # epochs=30
                    callbacks=[early_stopping_callback]) #,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
