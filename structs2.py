import keras
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Conv1D
from keras.models import Sequential
from keras.models import load_model
import keras.backend as K
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time
import keras.utils as k_



def isqrt(n):
    x = n
    y = (x + 1) / 2
    while y < x:
        x = y
        y = (x + n / x) / 2
    return x


def calcDimensions(n, d, s,o):
	N_ = 1+ float(n-d)/s
	N_int = 1+ (n-d)/s
	print(o,N_,N_, o*N_*N_)
	return N_int

def calcDimensions_inv(n, d, s, o):
	N_ = s*(n - 1) + d 
	print(o,N_,N_, o*N_*N_)
	return N_ 


def print_layers(model):
	for layer in model.layers:
	    print(layer.output_shape)
	print('\n')	


def layers6(input_shape):
	# N = 4
	model = Sequential()

	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'sigmoid'))

	model.add(Flatten())

	model.add(Dense(16384,activation='sigmoid'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S6.h5'



def diff4(input_shape):
	# N = 4

	init_ = keras.initializers.RandomUniform(minval=.1, maxval=.15) #.15,.3

	model = Sequential()

	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear',
	 init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear',
		init = init_))

	model.add(Reshape(input_shape))


	print_layers(model)

	def myLoss(y_true, y_pred):
		return K.mean((y_true-y_pred)**2) + 1./K.mean(y_pred)
	k_.get_custom_objects().update({"myLoss":myLoss})

	# SSIM, PSNR, changed lr from .01
	model.compile(loss = "myLoss",
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S4.h5'



def diff5(input_shape):
	# N = 4

	init_ = keras.initializers.RandomUniform(minval=.1, maxval=.15) #.15,.3

	model = Sequential()

	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear',
	 init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear',
		init = init_))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear',
		init = init_))

	model.add(Reshape(input_shape))


	print_layers(model)

	def myLoss2(y_true, y_pred):
		return K.mean((y_true-y_pred)**2) + .000003/K.mean(K.abs(y_pred))
	k_.get_custom_objects().update({"myLoss2":myLoss2})

	# SSIM, PSNR, changed lr from .01
	model.compile(loss = "myLoss2",
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S5.h5'



def layers4(input_shape):
	# N = 4
	model = Sequential()

	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR, changed lr from .01
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S4.h5'



def layers2(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S2.h5'


def layers3(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S3.h5'


def layers5(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))



	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.001),
		metrics = ['mae'])

	return model, 'S5.h5'


def layers6(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.001),
		metrics = ['mae'])

	return model, 'S6.h5'



def layers7(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(1024, kernel_size = 1024, strides = 1024, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.001),
		metrics = ['mae'])

	return model, 'S7.h5'


def layers8(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(1024, kernel_size = 1024, strides = 1024, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))# added this

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))


	model.add(Conv1D(2048, kernel_size = 2048, strides = 2048, activation = 'linear')) # and this

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'relu'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'sigmoid'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.001),
		metrics = ['mae'])

	return model, 'S8.h5'



def layers9(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(1024, kernel_size = 1024, strides = 1024, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))# added this

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))


	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear')) # and this

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))


	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S9.h5'


def layers10(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(1024, kernel_size = 1024, strides = 1024, activation = 'linear'))

	model.add(Reshape((16384,1)))



	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear'))

	model.add(Reshape((16384,1)))


	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))# added this

	model.add(Reshape((16384,1)))



	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear')) # and this

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))



	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape(input_shape))


	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S10.h5'


def layers11(input_shape):
	S = Sequential()

	model = load_model("S11_.h5")

	for layer in model.layers:
		S.add(layer)

	S.add(Flatten(input_shape = input_shape, name = "flatten_new2"))

	S.add(Reshape((16384,1)))

	S.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'sigmoid'))

	S.add(Reshape(input_shape))

	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	return model, 'S11.h5'


def layers0(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(1024, kernel_size = 1024, strides = 1024, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))

	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.0001),
		metrics = ['mae'])

	return model, 'S0.h5'


def layers1(input_shape):
	# N = 4
	model = Sequential()
	model.add(Flatten(input_shape = input_shape))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(32, kernel_size = 32, strides = 32, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(1024, kernel_size = 1024, strides = 1024, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(8, kernel_size = 8, strides = 8, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(64, kernel_size = 64, strides = 64, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4096, kernel_size = 4096, strides = 4096, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(128, kernel_size = 128, strides = 128, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(4, kernel_size = 4, strides = 4, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(256, kernel_size = 256, strides = 256, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(512, kernel_size = 512, strides = 512, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(16, kernel_size = 16, strides = 16, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2048, kernel_size = 2048, strides = 2048, activation = 'linear'))

	model.add(Reshape((16384,1)))

	model.add(Conv1D(2, kernel_size = 2, strides = 2, activation = 'linear'))

	model.add(Reshape(input_shape))

	print_layers(model)

	# SSIM, PSNR
	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.0001),
		metrics = ['mae'])

	return model, 'S1.h5'


def oneD(input_shape):

	model = Sequential()

	model.add(Flatten(input_shape = input_shape))

	model.add(Dense(16384, activation = 'relu'))

	#model.add(Dense(16384, activation = 'sigmoid'))

	#model.add(Dense(16384, activation = 'linear'))

	#model.add(Dense(16384, activation = 'relu'))

	model.add(Dense(16384, activation = 'linear'))

	model.add(Reshape(input_shape))

	model.compile(loss = keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adam(lr= 0.01),
		metrics = ['mae'])

	print_layers(model)

	return model, 'oneD.h5'