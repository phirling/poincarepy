import numpy as np

G_grav = 4.299581e04 # kpc * (km/s)^2 / 10^10 M_sun

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
        return [y[2],
                y[3],
                F[0],
                F[1]]
    def info(self):
        s = "Empty Potential"
        return(s)
    def __add__(self,otherpot):
        return SummedPotential(self,otherpot)


class LogarithmicPotential(Potential):
    def __init__(self,v0,rc,q):
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
        return("Logarithmic potential: v0 = {:.1f}, rc = {:.1f}, q  = {:.1f}".format(self.v0,self.rc,self.q))
    
    # -- Convenience methods (specific) -- #
    def maxval_x(self,E):
        return np.sqrt(np.exp(2*E/self.v0**2)-self.rc**2)
    def ydot(self,E,x,xdot):
        ED = 2*(E-self.phi([x,0])) - xdot**2
        if ED < 0:
            return None
        else:
            return np.sqrt(ED)

class PointMassPotential(Potential):
    def __init__(self,M):
        self.M = M

    def phi(self,y):
        return -G_grav*self.M/np.sqrt(y[0]**2 + y[1]**2)
    
    def accel(self,y):
        ff = -G_grav*self.M/((y[0]**2 + y[1]**2)**1.5)
        return np.array([y[0]*ff,y[1]*ff])

    def maxval_x(self,E):
        return G_grav*self.M/E

class HomospherePotential(Potential):
    """Potential of a homogeneous sphere
    """
    def __init__(self,a,M):
        self.rho = 3*M/(4*np.pi*a**3) 
        self.a = a
        self.M = M
        self._ff = -4*np.pi*G_grav*self.rho/3
        self._a2 = a*a
    def phi(self, y):
        r = np.asarray(np.sqrt(y[0]**2 + y[1]**2))
        return np.where(r<=self.a,-2*np.pi*G_grav*self.rho*(self._a2 - r**2/3),-4*np.pi*G_grav*self.rho*self.a**3 / (3*r))
    def accel(self, y):
        r2 = np.asarray(y[0]**2 + y[1]**2)
        return np.where(r2<=self._a2,self._ff*y[0:2],-G_grav*self.M/r2**1.5*y[0:2])
        if r2<=self._a2:
            return self._ff*y[0:2]
        else:
            return -G_grav*self.M/r2**1.5*y[0:2]
    def info(self):
        return("Homosphere potential: a = {:.1f}, M = {:.1e}$".format(self.a,self.M))

class zRotation(Potential):
    """
    Coriolis: -2*Omega x V
    Centrifugal: -Omega x (Omega x X)
    """
    def __init__(self,omega):
        self.omega = omega
    
    def accel(self, y):
        coriolis = -2. * np.array([-self.omega*y[3],self.omega*y[2]])
        centrif  = self.omega**2 * np.array([y[0],y[1]])
        return coriolis + centrif
    def phi(self, y):
        return -0.5*self.omega**2 * (y[0]**2 + y[1]**2)
    def info(self):
        return "z-axis Rotation: omega = {:.1f}".format(self.omega)

class SummedPotential(Potential):
    def __init__(self,pot1: Potential,pot2: Potential):
        self.pot1 = pot1
        self.pot2 = pot2
    def accel(self, y):
        return self.pot1.accel(y) + self.pot2.accel(y)
    def phi(self, y):
        return self.pot1.phi(y) + self.pot2.phi(y)
    def info(self):
        s = "Sum of potentials:\n  * " + self.pot1.info() + "\n  * " + self.pot2.info()
        return(s)