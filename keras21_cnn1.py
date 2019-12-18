from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(3, (2,2), padding='same',  # padding은 가장자리에 빈공간은 채워넣어서 가장자리쪽의 손실을 줄여줌
                input_shape= (5, 5, 1)))    # feature (5, 5, 3)
model.add(Conv2D(4,(2,2)))  # (4, 4, 4)
model.add(Conv2D(16,(2,2))) # (3, 3, 16)
# model.add(MaxPooling2D(3,3))
model.add(Conv2D(8,(2,2)))  # (2, 2, 8) Conv2D(filter(kernel_size(자르는 단위)), (stride=이동하는 단위)))
model.add(Flatten())    # Dense 모델에 들어갈수 있게 펴준다 다차원 -> 1차원으로 27 * 27 * 7 (None, 32)
model.add(Dense(10))    # (None, 10)
model.add(Dense(10))


model.summary()