import numpy as np
import matplotlib.pyplot as plt

#G_grav = 4.299581e04 # kpc * (km/s)^2 / 10^10 M_sun
G_grav = 1
"""Gravitational constant, set to one by default. Changing this value allows to use different unit systems."""

class Potential:
    """Base class of a physical potential

    Each potential must redefine at least 3 methods:
    phi: The potential at given phase space position
    accel: The acceleration at given phase space position
    info: String containing information about the specific potential

    accel is used to compute the RHS of the equation
    dy/dt = RHS(t,y), which is solved by the integrator
    """
    def __init__(self):
        pass
    
    def phi(self,y):
        pass
    def accel(self,y):
        pass
    def RHS(self,t,y):
        F = self.accel(y)
        return np.array([
                y[2],
                y[3],
                F[0],
                F[1]])
    def info(self):
        s = "Empty Potential"
        return(s)
    def plot_x(self,x0,x1,y=0,Npoints=100,ax=None):
        xrange = np.linspace(x0,x1,Npoints)
        if ax is None:
            return plt.plot(xrange,self.phi([xrange,np.zeros_like(xrange)]))
        else:
            return ax.plot(xrange,self.phi([xrange,np.zeros_like(xrange)]))
    def plotcontour(self,x0,x1,y0,y1,Npoints=100,levels=20,cmap='viridis',ax=None):
        xrange = np.linspace(x0,x1,Npoints)
        yrange = np.linspace(y0,y1,Npoints)
        X,Y = np.meshgrid(xrange,yrange)
        Z = self.phi([X,Y])
        if ax is None:
            return plt.contourf(X,Y,Z,levels=levels,cmap=cmap)
        else:
            return ax.contourf(X,Y,Z)


####### Concrete Potentials #######

class LogarithmicPotential(Potential):
    def __init__(self,v0=10.,rc=1.,q=0.8):
        self.type = 'log'
        self.v0 = v0
        self.rc = rc
        self.q = q
    def phi(self,y):
        return 0.5*self.v0**2 * np.log(self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
    def accel(self,y):
        A = - self.v0**2 / (self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
        ax = A*y[0]
        ay = A*y[1]/(self.q**2)
        return np.array([ax,ay])
    def info(self):
        return("Logarithmic potential: v0 = {:.1f}, rc = {:.1f}, q  = {:.2f}".format(self.v0,self.rc,self.q))

class HomospherePotential(Potential):
    """Potential of a homogeneous sphere
    """
    def __init__(self,a=1.,M=1):
        self.type = 'homosphere'
        self.rho = 3*M/(4*np.pi*a**3) 
        self.a = a
        self.M = M
        self._ff = -4*np.pi*G_grav*self.rho/3
        self._a2 = a*a
    def phi(self, y):
        r = np.asarray(np.sqrt(y[0]**2 + y[1]**2))
        return np.where(r<=self.a,-2*np.pi*G_grav*self.rho*(self._a2 - r**2/3),
                        -4*np.pi*G_grav*self.rho*self.a**3 / (3*r))
    def accel(self, y):
        r2 = np.asarray(y[0]**2 + y[1]**2)
        return np.where(r2<=self._a2,self._ff*y[0:2],-G_grav*self.M/r2**1.5*y[0:2])
    def info(self):
        return("Homosphere potential: a = {:.1f}, M = {:.1e}$".format(self.a,self.M))

class PlummerPotential(Potential):
    def __init__(self,a=5.,M=3e3):
        self.type = 'plummer'
        self.a2=a*a
        self.M=M
    def phi(self,y):
        return -G_grav*self.M/np.sqrt(self.a2 + y[0]**2 + y[1]**2)
    def accel(self,y):
        ff = -G_grav*self.M/((y[0]**2 + y[1]**2 + self.a2)**1.5)
        return np.array([y[0]*ff,y[1]*ff])
    def info(self):
        return("Plummer potential: a = {:.1f}, M = {:.1e}$".format(np.sqrt(self.a2),self.M))
class zRotation(Potential):
    """
    Coriolis: -2*Omega x V
    Centrifugal: -Omega x (Omega x X)
    """
    def __init__(self,omega):
        self.type = 'zrotation'
        self.omega = omega
    def accel(self, y):
        coriolis = -2. * np.array([-self.omega*y[3],self.omega*y[2]])
        centrif  = self.omega**2 * np.array([y[0],y[1]])
        return coriolis + centrif
    def phi(self, y):
        return -0.5*self.omega**2 * (y[0]**2 + y[1]**2)
    def info(self):
        return "z-axis Rotation: omega = {:.1f}".format(self.omega)

class CombinedPotential(Potential):
    def __init__(self,*pots: Potential):
        self.type = 'combined'
        self.potentials = self.manage_pots(pots)
    def manage_pots(self,given_potentials):
        pots = []
        for p in given_potentials:
            if p.type == 'combined':
                pots.extend(p.potentials)
            else:
                pots.append(p)
        return pots
    def phi(self,y):
        return sum([p.phi(y) for p in self.potentials ])
    def accel(self, y):
        return sum([p.accel(y) for p in self.potentials ])
    def info(self):
        s1 = "-- Combined potential of --\n"
        s = ""
        for p in self.potentials:
            s += p.info()
            s += "\n"
        return s1 + s

class EffectiveLogarithmic_cylindrical(Potential):
    def __init__(self,v0=10.,rc=1.,q=0.8,Lz=0.):
        self.type = 'log'
        self.v0 = v0
        self.rc = rc
        self.q = q
        self.Lz = Lz
    def phi(self,y):
        return (0.5*self.v0**2 * np.log(self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
                + self.Lz**2/(2*y[0]**2))
    def accel(self,y):
        ar = - self.v0**2 / (self.rc**2 + y[0]**2 + y[1]**2/self.q**2)*y[0] + self.Lz**2/y[0]**3
        az = - self.v0**2 / (self.rc**2 + y[0]**2 + y[1]**2/self.q**2)*y[1]/self.q**2
        return np.array([ar,az])
    def info(self):
        return("Effective Logarithmic potential (r,z): v0 = {:.1f}, rc = {:.1f}, q  = {:.1f}, Lz = {:.1f}".format(self.v0,self.rc,self.q,self.Lz))

class PointMassPotential(Potential):
    def __init__(self,M=1e3):
        self.type = 'pointmass'
        self.M = M
    def phi(self,y):
        return -G_grav*self.M/np.sqrt(y[0]**2 + y[1]**2)
    
    def accel(self,y):
        ff = -G_grav*self.M/((y[0]**2 + y[1]**2)**1.5)
        return np.array([y[0]*ff,y[1]*ff])

    def maxval_x(self,E):
        return G_grav*self.M/E