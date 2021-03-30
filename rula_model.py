from keras import Input, metrics
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D, Convolution2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from keras.layers import Add, ZeroPadding2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, LeakyReLU, UpSampling2D
from keras.models import Model, Sequential, load_model, model_from_json
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

def posture_estimation_model_func():

  # Instantiation
  Posture_Net = Sequential()

  #1st Convolutional Layer
  Posture_Net.add(Conv2D(filters=96, input_shape=(128,128,1), kernel_size=(11,11), strides=(4,4), padding='same'))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))
  Posture_Net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

  #2nd Convolutional Layer
  Posture_Net.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))
  Posture_Net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

  #3rd Convolutional Layer
  Posture_Net.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))

  #4th Convolutional Layer
  Posture_Net.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))

  #5th Convolutional Layer
  Posture_Net.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))

  Posture_Net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

  #Passing it to a Fully Connected layer
  Posture_Net.add(Flatten())
  # 1st Fully Connected Layer
  Posture_Net.add(Dense(4096))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))
  # Add Dropout to prevent overfitting
  Posture_Net.add(Dropout(0.4))

  #2nd Fully Connected Layer
  Posture_Net.add(Dense(4096))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))
  #Add Dropout
  Posture_Net.add(Dropout(0.4))

  #3rd Fully Connected Layer
  Posture_Net.add(Dense(1000))
  Posture_Net.add(BatchNormalization())
  Posture_Net.add(Activation('relu'))
  #Add Dropout
  Posture_Net.add(Dropout(0.4))

  #Output Layer
  Posture_Net.add(Dense(34))
  Posture_Net.add(BatchNormalization())

  #Model Summary
  #Posture_Net.summary()
  return Posture_Net

