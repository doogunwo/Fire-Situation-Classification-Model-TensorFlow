from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

# 카테고리 지정하기
categories = ["Fire","Non_Fire","Smoke"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64
image_h = 64
# 데이터 열기
X_train, X_test, y_train, y_test = np.load("data.npy",allow_pickle=True)
# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float") / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
print(model.summary())

# 학습 완료된 모델 저장
Bs = 32
Select_epochs =100
model.fit(X_train, y_train, batch_size=Bs, epochs=Select_epochs)
model_name  = "model/model_Bs="+str(Bs)+"_Se"+str(Select_epochs)+".h5"
model.save(model_name)

#모델 평가하기
score = model.evaluate(X_test,y_test)

print('loss=',score[0])
print('acc=',score[1])
