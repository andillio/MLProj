import keras
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time
import structs2 as s

input_shape = (4,4,1024)

train_iters = 1000
batch_size = 10
epochs_ = 20


save_ = 1
struct = s.layers11
save_file = struct(input_shape)[1]

load_ = 0
load_file = save_file

data_direct = "../data_compressed_T1/"
truth_direct = "../data_compressed_T5/"



def remaining(done, total, start):
	Dt = time.time() - start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	r = T
	hrs = int(r)/(60*60)
	mins = int(r%(60*60))/(60)
	s = int(r%60)
	return (hrs, mins, s)


def loadData(batch_size):
	batch_shape = (batch_size, input_shape[0], input_shape[1], input_shape[2])
	X = np.zeros(batch_shape)
	Y = np.zeros(batch_shape)

	Xfiles = os.listdir(data_direct)
	Yfiles = os.listdir(truth_direct)

	found = 0
	while(found < batch_size):
		try:
			i = np.random.randint(0, np.min([len(Xfiles), len(Yfiles)]))
			file_ = Xfiles[i]
			dataPoint = file_.split(".")[0]

			X[found] = np.load(data_direct + dataPoint + ".1.npy")
			Y[found] = np.load(truth_direct + dataPoint + ".5.npy")
			found += 1
		except IOError:
			pass

	return X, Y



if __name__ == "__main__":
	start = time.time()
	if load_:
		model = load_model(load_file) 
	else:
		model = struct(input_shape)[0]

	for i in range(train_iters):
		X,Y = loadData(batch_size)
		X_v, Y_v = loadData(batch_size/10)

		model.fit(X,Y,epochs=epochs_,verbose=2,validation_data=(X_v,Y_v))

		if save_:
			model.save(save_file)
		print(('%i hrs, %i mins, %i s remaining.' %remaining(i+1., train_iters, start)), save_file)

	print("done")
