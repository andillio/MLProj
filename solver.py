import numpy as np
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata
import os
import pdb
import copy

filename = 'IC/IC/data'
filetar  = 'IC/Sim/data'

x = np.linspace(0,1,257)
y = np.linspace(0,1,257)

def MinMod(x,y,z):
    return 1/4.*np.abs(np.sign(x) + np.sign(y))*(np.sign(x) + np.sign(z))*np.min(np.abs(np.array([x,y,z])), axis = 0)

def PLM(c, axis, side, theta = 1.5, on=True):
    if side == 'L':
        x = theta*(c - np.roll(c,1, axis = axis))
        y = 0.5*(np.roll(c,-1, axis = axis) - np.roll(c,1, axis = axis))
        z = theta*(np.roll(c,-1, axis = axis) - c)
        plm = c + on*0.5*MinMod(x,y,z) #ASSUMES UNIFORM GRID
    elif side == 'R':
        x = theta*(np.roll(c,-1,axis = axis) - c)
        y = 0.5*(np.roll(c,-2, axis = axis) - c)
        z = theta*(np.roll(c,-2, axis = axis) - np.roll(c,-1, axis = axis))
        plm = np.roll(c,-1,axis = axis) - on*0.5*MinMod(x,y,z)#ASSUMES UNIFORM GRID
    else:
        print('PLM SIDE NOT SPECIFIED')
    return plm

def HydroInitialization(x, y, rho, vx, vy, g, ied):
    #Generate Grid
    Grid = np.zeros((3,4,len(x)-1,len(y)-1)) #Grid[0] == U, Grid[1] == F, Grid[2] == G

    #Conserved Variable U
    Grid[0][0] = rho
    Grid[0][1] = rho*vx
    Grid[0][2] = rho*vy
    Grid[0][3] = rho*(ied + (vx**2 + vy**2)/2)

    #Flux F
    Grid[1][0] = Grid[0][1]
    Grid[1][1] = Grid[0][1]*vx + (g-1)*ied*Grid[0][0]
    Grid[1][2] = rho*vx*vy
    Grid[1][3] = (rho*(ied + (vx**2 + vy**2)/2))*vx +(g-1)*ied*Grid[0][0]*vx

    #Flux G
    Grid[2][0] = Grid[0][2]
    Grid[2][1] = Grid[1][2]
    Grid[2][2] = Grid[0][2]*vy + (g-1)*ied*Grid[0][0]
    Grid[2][3] = (rho*(ied  + (vx**2 + vy**2)/2))*vy + (g-1)*ied*Grid[0][0]*vy

    return Grid

class Solver:
    def __init__(self, x, y, rho, vx, vy, g = 2, ied = np.zeros((len(x)-1,len(y)-1))+10):
        self.x = x
        self.y = y
        self.g = g
        self.Grid = HydroInitialization(x, y, rho, vx, vy, self.g, ied)

    def IG_eos(self):
        ied = self.Grid[0][3]/self.Grid[0][0] - 1/2.*(self.Grid[0][1]**2 + self.Grid[0][2]**2)/self.Grid[0][0]**2
        return (self.g-1)*ied*self.Grid[0][0]

    def L(self,yo):
        vx = np.divide(self.Grid[0][1], self.Grid[0][0])
        vy = np.divide(self.Grid[0][2], self.Grid[0][0])
        c = np.sqrt(self.g*self.IG_eos()/self.Grid[0][0])
        on = yo #PLM

        #Interface Flux F
        ap1 = np.amax(np.array([np.zeros(vx.shape),  vx[:,:] + c[:,:],  np.roll(vx[:,:] + c[:,:],-1,axis=0)]),axis=0)
        am1 = np.amax(np.array([np.zeros(vx.shape), -vx[:,:] + c[:,:], -np.roll(vx[:,:] - c[:,:],-1,axis=0)]),axis=0)
        ap1 = np.amax(np.array([np.zeros(vx.shape), PLM( vx[:,:] + c[:,:],0,'L',on=on), PLM( vx[:,:] + c[:,:],0,'R',on=on)]),axis=0)
        am1 = np.amax(np.array([np.zeros(vx.shape), PLM(-vx[:,:] + c[:,:],0,'L',on=on), PLM(-vx[:,:] + c[:,:],0,'R',on=on)]),axis=0)

        self.IF_F = np.zeros(np.array(self.Grid[1].shape) + np.array([0,1,0]))
        for i in range(self.IF_F.shape[0]):
            self.IF_F[i,1:,:] = (ap1*PLM(self.Grid[1][i,:,:],0,'L',on=on) + am1*PLM(self.Grid[1][i,:,:],0,'R',on=on) - ap1*am1*(PLM(self.Grid[0][i,:,:],0,'R',on=on)-PLM(self.Grid[0][i,:,:],0,'L',on=on)))/(ap1 + am1)
            self.IF_F[i,0,:] = self.IF_F[i,-1,:]

        #Interface Flux G
        ap2 = np.amax(np.array([np.zeros(vy.shape),  vy[:,:] + c[:,:],  np.roll(vy[:,:] + c[:,:],-1, axis=1)]),axis=0)
        am2 = np.amax(np.array([np.zeros(vy.shape), -vy[:,:] + c[:,:], -np.roll(vy[:,:] - c[:,:],-1, axis=1)]),axis=0)
        ap2 = np.amax(np.array([np.zeros(vy.shape), PLM( vy[:,:] + c[:,:],1,'L',on=on), PLM( vy[:,:] + c[:,:],1,'R',on=on)]),axis=0)
        am2 = np.amax(np.array([np.zeros(vy.shape), PLM(-vy[:,:] + c[:,:],1,'L',on=on), PLM(-vy[:,:] + c[:,:],1,'R',on=on)]),axis=0)
        self.IF_G = np.zeros(np.array(self.Grid[2].shape) + np.array([0,0,1]))
        for i in range(self.IF_G.shape[0]):
            self.IF_G[i,:,1:] = (ap2*PLM(self.Grid[2][i,:,:],1,'L',on=on) + am2*PLM(self.Grid[2][i,:,:],1,'R',on=on) - ap2*am2*(PLM(self.Grid[0][i,:,:],1,'R',on=on)-PLM(self.Grid[0][i,:,:],1,'L',on=on)))/(ap2 + am2)
            self.IF_G[i,:,0] = self.IF_G[i,:,-1]
        self.MAX = np.amax(np.array([ap1,am1,ap2,am2]))

    def FluxUpdate(self):
        vx = np.divide(self.Grid[0][1], self.Grid[0][0])
        vy = np.divide(self.Grid[0][2], self.Grid[0][0])

        self.Grid[1][0] = self.Grid[0][1]
        self.Grid[1][1] = np.multiply(self.Grid[0][0], np.power(vx, 2))+ self.IG_eos()
        self.Grid[1][2] = np.multiply(np.multiply(self.Grid[0][0], vx), vy)
        self.Grid[1][3] = np.multiply((self.Grid[0][3]+ self.IG_eos()), vx)

        self.Grid[2][0] = self.Grid[0][2]
        self.Grid[2][1] = self.Grid[1][2]
        self.Grid[2][2] = np.multiply(self.Grid[0][0], np.power(vy, 2)) + self.IG_eos()
        self.Grid[2][3] = np.multiply((self.Grid[0][3] + self.IG_eos()), vy)

    def TimeEvolver(self,T,save=True,fname='HydroSimKHI',yo=1,rk3=True):             
        t = 0.
        count = 0
        frames = 10
        time = T/frames
        DT = time
        n = 0
        newmax = np.amax(self.Grid[0][0])
        newmin = np.amin(self.Grid[0][0])
        while t < T:
            self.L(yo)
            dt = np.amin(np.array([T - t, time - t, 0.5*np.amin(np.append(self.x[1:]-self.x[:-1],self.y[1:]-self.y[:-1]))/self.MAX]))
            #RK3 Time Evolution
            U_n = copy.deepcopy(self.Grid[0])

            self.dUdt = -(self.IF_F[:,1:,:]-self.IF_F[:,:-1,:])/(self.x[1:]-self.x[:-1]) -(self.IF_G[:,:,1:]-self.IF_G[:,:,:-1])/(self.y[1:]-self.y[:-1]) 
            self.Grid[0] = self.Grid[0] + dt*self.dUdt

            self.FluxUpdate()
            TEon = rk3
            if TEon:
                self.L(yo)
                self.dUdt = -(self.IF_F[:,1:,:]-self.IF_F[:,:-1,:])/(self.x[1:]-self.x[:-1]) -(self.IF_G[:,:,1:]-self.IF_G[:,:,:-1])/(self.y[1:]-self.y[:-1]) 
                self.Grid[0] = 3/4.*U_n + 1/4.*self.Grid[0] + 1/4.*dt*self.dUdt
                self.FluxUpdate()

                self.L(yo)
                self.dUdt = -(self.IF_F[:,1:,:]-self.IF_F[:,:-1,:])/(self.x[1:]-self.x[:-1]) -(self.IF_G[:,:,1:]-self.IF_G[:,:,:-1])/(self.y[1:]-self.y[:-1]) 
                self.Grid[0] = 1/3.*U_n + 2/3.*self.Grid[0] + 2/3.*dt*self.dUdt
                self.FluxUpdate()
            #M = np.sum(self.Grid[0][0]*(self.x[1:]-self.x[:-1])*(self.y[1:]-self.y[:-1]))
            #E = np.sum(self.Grid[0][3])
            print(dt)
            t = t + dt
            if t in [time, T]:
                time = time + DT
                n = n + 1
                #plt.imshow(self.Grid[0][0],interpolation='None')
                #plt.colorbar()
                #plt.clim(newmin,newmax)
                #plt.xlabel('x')
                #plt.xlabel('y')
                #plt.title('Density')
                if save:
		    X = self.Grid[0]
                    X[1] /= X[0]
                    X[2] /= X[0]
                    X[1] += 1
                    X[2] += 1
                    np.save(fname+'.'+str(n)+'.npy',X)
                    X[1] += -1
                    X[2] += -1
                    X[1] *= X[0]
                    X[2] *= X[0]
            count = count + 1
            print(t)
#KH Instability, Higher Order
#Define Coordinate Grid (Interfaces)
x = np.linspace(0,1,257)
y = np.linspace(0,1,257)
#Kelvin Helmholtz



#Number of things to simulate
Nsim = 100
for i in range(Nsim):#Initial Conditions
	done = np.loadtxt('done.txt')
	j = int(np.amin(done))
	while j in done or j < 0:
		j = j + 1
	done = np.append(j,done)
	np.savetxt('done.txt', done,fmt='%d')

	X = np.load(filename+str(j)+'.npy')
	X = X.reshape((4,256,256))
	rho = X[0]
	vx  = X[1]-1
	vy  = X[2]-1
	ied = X[3]/rho - 1./2*(vx**2 + vy**2)

	g = 5./3
	sim = Solver(x,y,rho,vx,vy, g = g,ied=ied)
	sim.TimeEvolver(1.,fname=filetar+str(j),yo=1,rk3=True) #yo turns on PLM
