import numpy as np 
import scipy.fftpack as sp
import matplotlib.pyplot as plt
import time
import sys

IC_2_gen = 2000 # how many initial conditions should be generated

N = 256 # number of cells in grid
L = 1. # side length of grid

plot_ = 0 # plot fields (for inspection/debugging)
output_ = 1 # should text files be output?
eta = 1 # want it to print the estimated time remaining to the console?
fft_ = 1

# field stuff
# the mean is the approximate average value 
# the max_var is the scale of the deviations on top of the mean
# i.e. f(x,y) about [1 - max_var, 1 + max_var]

data_direct = "data/"
starting_index = 0

# density 
mean_density = 1.
max_var_density = .1
density_file_name = "density"

# vx
mean_vx = 1.
max_var_vx = .1
vx_file_name = "vx"

# vy
mean_vy = 1.
max_var_vy = .1
vy_file_name = "vy"

# pressure 
mean_pressure = 1.
max_var_pressure = .1
pressure_file_name = "pressure"

# power spectrum stuff
# - there will be no structure smaller than dx*fac
# - the power spectrum is a random gaussian in k space with mean choosen log uniform
#   between 1./N/L and (2pi)*N/L/fac with sigma chosen uniformly between range(k)/f_var_min
#   and range(k)/f_var_max. where range k is k_max - k_min
fac = 10.
f_var_min = 200.
f_var_max = 50.


# returns the a grid (G) in k space where each element G_ij = magnitude([k_i,k_j])
def get_k_mag():
	dx = L/N

	kx = 2*np.pi*sp.fftfreq(N, d = dx)

	kx, ky = np.meshgrid(kx, kx)

	return np.sqrt(kx**2 + ky**2)


# returns a gaussian random variable with given mean and standard deviation
def gaussVar(mu, var):

	return np.random.normal(mu, np.sqrt(var))

# the power spectrum class
class Powr(object):
	def __init__(self):

		dx = L/N

		kx = 2*np.pi*sp.fftfreq(N, d = dx)

		kx = np.abs(kx)[:N/2]

		kmin = np.min(kx)
		kmax = np.max(kx)
		krange = kmax - kmin

		# basically stores information for a random gaussian distributed power spectrum
		self.mu = 0*np.exp(np.random.uniform(np.log(1./N/L), np.log(kmax/fac)))
		self.var = np.random.uniform(krange/f_var_min, krange/f_var_max)**2
		self.N = np.sqrt(1./(2*np.pi*self.var))

	# returns the variances associated with given k_magnitudes. i.e. P(|k|)
	def getP(self,k_mag):
		return self.N*np.exp(-.5*(self.mu - k_mag)**2 / self.var)


# returns a random field with power spectrum P
def randomField(P):
	f = np.zeros((N,N)) +0j

	k_mag = get_k_mag()

	mu = np.zeros((N/2-1,N/2-1))
	var = P.getP(k_mag[1:N/2,1:N/2])
	
	A = gaussVar(mu, var)
	phase = np.exp(1j*np.random.uniform(0.,np.pi*2., (N/2-1,N/2-1)))

	f[1:N/2,1:N/2] = A*phase
	f[N/2+1:,N/2+1:] = np.flip(np.flip(np.conj(A*phase),0),1)

	var = P.getP(k_mag[1:N/2,N/2+1:])
	A = gaussVar(mu, var)
	phase = np.exp(1j*np.random.uniform(0.,np.pi*2., (N/2-1,N/2-1)))

	f[1:N/2,N/2+1:] = A*phase
	f[N/2+1:,1:N/2] = np.flip(np.flip(np.conj(A*phase),0),1)

	return sp.ifft2(f).real


# normalizes the input field to have mean mu with variations of size 2*df
def normalize(f, mu, df):
	rval = np.zeros(np.shape(f))
	range_ = np.max(f) - np.min(f)
	return mu + (f - np.mean(f))*2.*df/range_ 


# outputs a text files containing the field f
def output(g, name_, i):
	f = g
	if fft_:
		f = sp.fft2(g)
	if output_:
		f_name = data_direct + name_ + str(i + starting_index) + ".txt"
		np.savetxt(f_name, f)


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

def repeat_print(string):
    sys.stdout.write('\r' +string)
    sys.stdout.flush()


if __name__ == "__main__":
	start = time.time()
	for i in range(IC_2_gen):
		P = Powr()
		density = normalize(randomField(P), mean_density, max_var_density)
		output(density, density_file_name, i)

		vx = normalize(randomField(P), mean_vx, max_var_vx)
		output(vx, vx_file_name, i)

		vy = normalize(randomField(P), mean_vy, max_var_vy)
		output(vy, vy_file_name, i)

		pressure = normalize(randomField(P), mean_pressure, max_var_pressure)
		output(pressure, pressure_file_name, i)

		if eta:
			repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(i+1, IC_2_gen, start)))

		if plot_:
			fig, axs = plt.subplots(2,2, figsize = (20,20))

			axs[0][0].contourf(density)
			axs[0][1].contourf(vx)
			axs[1][0].contourf(pressure)
			axs[1][1].contourf(vy)

			plt.show()
	print("\n\ndone\n")
