import os
import glob
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np

# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

img_width, img_height = 150, 150
train_data_dir = '~/Doucuments/Learning/train'
validation_data_dir = '~/Documents/Learning/validation'
nb_train_samples = 16*100
nb_validation_samples = 10
nb_epoch = 50
result_dir = 'results'

x_train = np.zeros((1600,2448,3264,3))
images = glob.glob(os.path.join('./train/extended/', "*.jpg"))
for i in range(len(images)):
    img = load_img(images[i])
    part = img_to_array(img)
    x_train[i] = img
    
x_test = x_train[0:10]    
    
x_train = cv2.resize(x_train,(32,32))
x_test = cv2.resize(x_test,(32,32))

x_train = x_train.reshape(1700, 3072)
x_test = x_test.reshape(1700, 3072)

y_train = np.ones(1700)
y_test = np.ones(10)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')








model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


#Learn
history=model.fit(x_train,y_train,batch_size=128,nb_epoch=10,verbose=1,validation_split=0.1)



#モデルと重みを保存
json_string=model.to_json()
open('cifar10_cnn.json',"w").write(json_string)
model.save_weights('cifar10_cnn.h5')

#モデルの表示
model.summary()




#評価
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])