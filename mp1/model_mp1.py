### This file includes the functions for building models

from keras.models import *
from keras.layers import *
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Activation, Dropout,Convolution2D, MaxPooling2D, Flatten,BatchNormalization,UpSampling2D
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def DenseModel(optimizer='sgd'):
    """ build model for Q3 Simple classification """
    model = Sequential([Dense(3, input_shape=(5184,), activation='softmax')])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def CNN_ClassModel(input_shape=(1, 72, 72),n_class=3,lr=0.001):
    """ build model for Q5 a more difficult classification problem """
    model=Sequential()
    model.add(Convolution2D(filters=16,nb_row=5,nb_col=5, 
    #                         kernel_size=5,strides=1,
                padding='same',data_format='channels_first',
                input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',data_format='channels_first'))
    model.add(Dropout(0.9))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    adam=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def CNN_RegModel(input_shape=(1,72,72),output_dim=6,lr=0.0008,model_path='model.h5'):
    """ build model for Q6 regression problem """
    model=Sequential()
    model.add(Convolution2D(filters=16,nb_row=5,nb_col=5, padding='same',data_format='channels_first',input_shape=input_shape))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',data_format='channels_first'))

    model.add(Convolution2D(filters=32,nb_row=5,nb_col=5, padding='same',data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',data_format='channels_first'))
    
    model.add(Convolution2D(filters=64,nb_row=5,nb_col=5, padding='same',data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',data_format='channels_first'))
   
    model.add(Convolution2D(filters=128,nb_row=5,nb_col=5, padding='same',data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',data_format='channels_first'))
    
    model.add(Flatten())
    model.add(Dense(5184))
    model.add(Activation('relu'))

    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('tanh'))

    adam=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(loss='mse',optimizer=adam,metrics=['acc']) 
    # define the checkpoint
    filepath = model_path
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # print(model.summary())
    return model, callbacks_list

def autoencoder(input_shape=(72,72,1),lr=0.001):
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), padding='same', data_format='channels_last', input_shape=input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))

    model.add(UpSampling2D(size=(2, 2), data_format='channels_last'))
    model.add(Convolution2D(32, (5, 5), padding='same', data_format='channels_last'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    # model4.add(UpSampling2D(size=(2, 2),data_format='channels_first'))
    model.add(Convolution2D(1, (5, 5), padding='same', data_format='channels_last'))
    model.add(Activation('sigmoid'))

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(optimizer=adam,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model
    
def hourglass(input_shape=(72,72,1),lr=0.001):
    """ build model for Q7 Image denoising """
    k_size = (3,3)
    input = Input(shape=input_shape)

    _x = Convolution2D(32, kernel_size=k_size, strides=(2, 2), data_format='channels_last',padding='same') (input)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)
    _x = Convolution2D(32, k_size, padding='same', data_format='channels_last')(_x)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)
    _x_branch = MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')(_x)
    
    _x = Convolution2D(128, k_size, padding='same', data_format='channels_last')(_x_branch)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)
    _x = Convolution2D(128, k_size, padding='same', data_format='channels_last')(_x)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)
    _x = MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last')(_x)
    
    _x = UpSampling2D(size=(2, 2), data_format='channels_last')(_x)
    _x = Convolution2D(64, k_size, padding='same', data_format='channels_last')(_x)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)
    _x = Convolution2D(64, k_size, padding='same', data_format='channels_last')(_x)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)

    _x = concatenate([_x,_x_branch],axis=3)
    
    _x = UpSampling2D(size=(2, 2), data_format='channels_last')(_x)
    _x = Convolution2D(32, k_size, padding='same', data_format='channels_last')(_x)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)
    
    _x = UpSampling2D(size=(2, 2), data_format='channels_last')(_x)
    _x = Convolution2D(32, k_size, padding='same', data_format='channels_last')(_x)
    _x = BatchNormalization(axis=3)(_x)
    _x = Activation('relu')(_x)

    _x = Convolution2D(1, k_size, padding='same', data_format='channels_last')(_x)
    _x = Activation('sigmoid')(_x)

    
    model = Model(inputs=input, outputs=_x)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(optimizer=adam,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model
