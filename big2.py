import keras
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from keras.models import Sequential
from keras.models import load_model
import numpy as np
#import tensorflow as tf
import os
#import matplotlib.pyplot as plt
import time
from keras import backend as K
import glob

# Not recommended; however this will correctly find the necessary contrib modules
from keras_contrib import *

from theano.sandbox.neighbours import images2neibs
from keras.backend import theano_backend as KTH


from keras.utils.generic_utils import get_custom_objects



print("Modules Loaded")


N = 256

batch_size = 128
epochs_ = 6
maew = 1.5
dsimw = 1.0
lr = 1e-3
trainiter = 1000

TrainOrig=True

save_ = 1
save_file = "big2.h5"
load_ = 1
load_file = save_file
predict_ = 0

input_shape = (4,N,N)

data_direct = "IC/IC/"


'''
from theano import function, config, shared, tensor
import numpy
import time


vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
'''

#from tensorflow.python.client import device_lib
#if 'GPU' in str(device_lib.list_local_devices()):
#	print("GPU ENABLED!")
#else:
#	print("CPU ENABLED!")
#assert(False)


def extract_image_patches(X, ksizes, strides, padding='valid', data_format='channels_first'):
    patch_size = ksizes[1]
    if padding == 'same':
        padding = 'ignore_borders'
    if data_format == 'channels_last':
        X = KTH.permute_dimensions(X, [0, 3, 1, 2])
    # Thanks to https://github.com/awentzonline for the help!
    batch, c, w, h = KTH.shape(X)
    xs = KTH.shape(X)
    num_rows = 1 + (xs[-2] - patch_size) // strides[1]
    num_cols = 1 + (xs[-1] - patch_size) // strides[1]
    num_channels = xs[-3]
    patches = images2neibs(X, ksizes, strides, padding)
    # Theano is sorting by channel
    patches = KTH.reshape(patches, (batch, num_channels, num_rows * num_cols, patch_size, patch_size))
    patches = KTH.permute_dimensions(patches, (0, 2, 1, 3, 4))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = KTH.reshape(patches, (batch, num_rows, num_cols, num_channels, patch_size, patch_size))
    if data_format == 'channels_last':
        patches = KTH.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


import keras_contrib.backend as KC
#from keras_contrib.losses import DSSIMObjective


class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=16, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.

        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
	self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = KC.backend()

    def __int_shape(self, x):
        return KC.int_shape(x) if self.backend == 'tensorflow' else KC.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = KC.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = KC.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

	stride = [self.kernel_size, self.kernel_size]
        patches_pred = extract_image_patches(y_pred, kernel, stride, 'valid', self.dim_ordering)
        patches_true = extract_image_patches(y_true, kernel, stride, 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = KC.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = KC.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = KC.mean(patches_true, axis=-1)
        u_pred = KC.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

	ssim = (2. * u_true * u_pred + self.c1) * (2. * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
	ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
	#return K.mean((1.0 - ssim) / 2.0)
	return K.mean((1.0 - ssim) /2.0)


def calcDimensions(n, d, s,o):
	N_ = 1+ float(n-d)/s
	N_int = 1+ (n-d)//s
	print(o,N_,N_)
	return N_int

def calcDimensions_inv(n, d, s, o):
	N_ = s*(n - 1) + d
	print(o, N_, N_)
	return N_


def print_layers(model):
	for layer in model.layers:
	    print(layer.output_shape)
	print('\n')


def makeModel():
	model = Sequential()
	
	
	################ encoder ##################



	d1, s1, o1 = 2,2,128
	clshape = [256,256,4]
	model.add(Conv2D(o1, kernel_size = (d1,d1), strides = (s1,s1),
	activation = 'relu', input_shape = clshape,trainable=TrainOrig))
	N1 = calcDimensions(N,d1,s1,o1)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d1, s1, o1 = 2,2,256
        model.add(Conv2D(o1, kernel_size = (d1,d1), strides = (s1,s1),
        activation = 'relu',trainable=TrainOrig))
        N1 = calcDimensions(N1,d1,s1,o1)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d2, s2, o2 = 2,2,256
	model.add(Conv2D(o2, kernel_size = (d2,d2), strides = (s2,s2),
		activation = 'relu',trainable=TrainOrig))
	N2 = calcDimensions(N1,d2,s2,o2)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))
	d2, s2, o2 = 2,2,512
	model.add(Conv2D(o2, kernel_size = (d2,d2), strides = (s2,s2),
		activation = 'relu',trainable=TrainOrig))
	N2 = calcDimensions(N2,d2,s2,o2)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d2, s2, o2 = 2,2,512
	model.add(Conv2D(o2, kernel_size = (d2,d2), strides = (s2,s2),
		activation = 'relu',trainable=TrainOrig))
	N2 = calcDimensions(N2,d2,s2,o2)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d2, s2, o2 = 2,2,1024
	model.add(Conv2D(o2, kernel_size = (d2,d2), strides = (s2,s2),
		activation = 'sigmoid',trainable=TrainOrig))
	N2 = calcDimensions(N2,d2,s2,o2)

	d2, s2, o2 = 2,2,2048
	model.add(Conv2D(o2, kernel_size = (d2,d2), strides = (s2,s2),
		activation = 'relu', name = 'big1', trainable=TrainOrig))
	N2 = calcDimensions(N2,d2,s2,o2)

	d2, s2, o2 = 2,2,4096
	model.add(Conv2D(o2, kernel_size = (d2,d2), strides = (s2,s2),
		activation = 'sigmoid', name = 'big2', trainable=TrainOrig))
	N2 = calcDimensions(N2,d2,s2,o2)


	###########################################

	############### decoder ###################


	d3, s3, o3 = 2, 2, 2048
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', name = 'bigT1', trainable=TrainOrig))
        N3 = calcDimensions_inv(N2, d3, s3, o3)

	d3, s3, o3 = 2, 2, 1024
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', name = 'bigT2', trainable=TrainOrig))
        N3 = calcDimensions_inv(N3, d3, s3, o3)

	d3, s3, o3 = 2, 2, 512
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', trainable=TrainOrig))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d3, s3, o3 = 2, 2, 512
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', trainable=TrainOrig))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d3, s3, o3 = 2, 2, 256
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', trainable=TrainOrig))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d3, s3, o3 = 2, 2, 256
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu',trainable=TrainOrig))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	d3, s3, o3 = 2, 2, 128
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu',trainable=TrainOrig))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))
	d3, s3, o3 = 2, 2, 64
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', name = 'bigg1'))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))
	d3, s3, o3 = 2, 2, 32
        model.add(Conv2DTranspose(o3,kernel_size = (d3,d3), strides = (s3,s3),
                activation = 'relu', name = 'bigg2'))
        N3 = calcDimensions_inv(N3, d3, s3, o3)
	#model.add(keras.layers.LeakyReLU(alpha=0.3))

	# layer 2
	d4, s4, o4 = 2, 2, 4
	model.add(Conv2D(o4,kernel_size = (d4,d4), strides = (s4,s4),
 		activation = 'linear', name = 'bigg3'))
 	N4 = calcDimensions_inv(N3, d4, s4, o4)



	return model



def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def remaining(done, total, start):
	Dt = time.time() - start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	r = T
	hrs = int(r)//(60*60)
	mins = int(r%(60*60))//(60)
	s = int(r%60)
	return (hrs, mins, s)


def loadData(size = 1000, size_ = batch_size):
	batch_shape = (size_, input_shape[1], input_shape[2], input_shape[0])
	X = np.zeros(batch_shape)
	Y = np.zeros(batch_shape)

	files = glob.glob(data_direct+"*")
	ran = np.random.randint(0,len(os.listdir(data_direct))-1, size=size_)
	for i in range(len(ran)):
		x = np.load(files[ran[i]])
		x = np.swapaxes(x,0,1)
		x = np.swapaxes(x,1,2)
		X[i] = x
		Y[i] = x

	return X, Y


if __name__ == "__main__":
	print("start")
	DSSIM1 = DSSIMObjective(kernel_size=8)
	def DSSIM(y_true,y_pred):
		return dsimw*DSSIM1(y_true, y_pred) + maew*K.mean(K.abs(y_true-y_pred))
	get_custom_objects().update({"DSSIM": DSSIM})


	if not predict_:
		if not load_:
			print('Model Generating')
			model = makeModel()
			model.load_weights('big.h5', by_name=True)
		if load_:
			print('Model Loading')
			model = makeModel()
			model.load_weights(load_file)
		print("Weights Locked n' Loaded")
		model.compile(loss = DSSIM,
		optimizer=keras.optimizers.Adam(lr=lr))#,
		#optimizer=keras.optimizers.SGD(lr=lr))
		#metrics = ['mae'])
		model.summary()

		for i in range(trainiter):

			print('Loading Data')
			X, Y = loadData(batch_size, batch_size)

			print('Data Loaded')
			model.fit(X,Y,epochs=epochs_,verbose=2, batch_size=1)
			x = X[0].reshape((1,256,256,4))
			Y = model.predict(x)
			np.save('Predict',Y)
        		np.save('Input',x)

			model.save(save_file)
			print("Saved Model")

	if predict_:
		model = load_model(load_file)
		print('Model Loaded')
		files = glob.glob(data_direct+"*")
		ran = np.random.randint(0,len(os.listdir(data_direct))-1)
		X_ = np.load(files[ran])
		X_ = np.swapaxes(X_,0,1)
		X_ = np.swapaxes(X_,1,2)
		X_ = X_.reshape((1,256,256,4))
		Y_ = model.predict(X_)
		np.save('Predict',Y_)
		np.save('Input',X_)

	print("done")
