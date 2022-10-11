import numpy as np

#G_grav = 4.299581e04 # kpc * (km/s)^2 / 10^10 M_sun
G_grav = 1

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
    def get_energyrange(self):
        pass
    def info(self):
        s = "Empty Potential"
        return(s)


class LogarithmicPotential(Potential):
    def __init__(self,v0=10.,rc=1.,q=0.8,zeropos = None):
        self.type = 'log'
        self.v0 = v0
        self.rc = rc
        self.q = q
        # Fix gauge
        if zeropos is None:
            self.gaugeparam = 1.0
        else:
            self.gaugeparam = 1./(rc**2 + zeropos[0]**2 + zeropos[1]**2/q**2)
    def phi(self,y):
        return 0.5*self.v0**2 * np.log(self.gaugeparam*(self.rc**2 + y[0]**2 + y[1]**2/self.q**2))
    def accel(self,y):
        A = - self.v0**2 / (self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
        ax = A*y[0]
        ay = A*y[1]/(self.q**2)
        return np.array([ax,ay])
    def get_energyrange(self):
        Emin = self.phi(np.zeros(4))
        Emax = np.inf
        return np.array([Emin,Emax])
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

class HomospherePotential(Potential):
    """Potential of a homogeneous sphere
    """
    def __init__(self,a=5.,M=1e3,zeropos=None):
        self.type = 'homosphere'
        self.rho = 3*M/(4*np.pi*a**3) 
        self.a = a
        self.M = M
        self._ff = -4*np.pi*G_grav*self.rho/3
        self._a2 = a*a
        if zeropos is None:
            self.gaugeparam = 0.0
        else:
            r0 = np.sqrt(zeropos[0]**2 + zeropos[1]**2)
            if r0 < a:
                self.gaugeparam = 2*np.pi*G_grav*self.rho*(a*a - r0**2/3.)
            else:
                self.gaugeparam = 4*np.pi*G_grav*self.rho*a**3/(3*r0)
    def phi(self, y):
        r = np.asarray(np.sqrt(y[0]**2 + y[1]**2))
        return np.where(r<=self.a,-2*np.pi*G_grav*self.rho*(self._a2 - r**2/3) + self.gaugeparam,
                        -4*np.pi*G_grav*self.rho*self.a**3 / (3*r) + self.gaugeparam)
    def accel(self, y):
        r2 = np.asarray(y[0]**2 + y[1]**2)
        return np.where(r2<=self._a2,self._ff*y[0:2],-G_grav*self.M/r2**1.5*y[0:2])
    def get_energyrange(self):
        Emin = self.phi(np.zeros(4))
        Emax = self.gaugeparam
        return np.array([Emin,Emax])
    def info(self):
        return("Homosphere potential: a = {:.1f}, M = {:.1e}$".format(self.a,self.M))

class zRotation(Potential):
    """
    Coriolis: -2*Omega x V
    Centrifugal: -Omega x (Omega x X)
    """
    def __init__(self,omega,zeropos=None):
        self.type = 'zrotation'
        self.omega = omega
        if zeropos is None:
            self.gaugeparam = 0.0
        else:
            self.gaugeparam = 0.5*omega**2*(zeropos[0]**2 + zeropos[1]**2)
    def accel(self, y):
        coriolis = -2. * np.array([-self.omega*y[3],self.omega*y[2]])
        centrif  = self.omega**2 * np.array([y[0],y[1]])
        return coriolis + centrif
    def phi(self, y):
        return -0.5*self.omega**2 * (y[0]**2 + y[1]**2) + self.gaugeparam
    def get_energyrange(self):
        Emin = -np.inf
        Emax = self.gaugeparam
        return np.array([Emin,Emax])
    def info(self):
        return "z-axis Rotation: omega = {:.1f}".format(self.omega)

"""### EXPERIMENTAL ###"""
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
    def get_energyrange(self):
        sub = np.array([p.get_energyrange() for p in self.potentials])
        Emax = np.amax(sub)
        Emin = np.amin(sub)
        return np.array([Emin,Emax])


class PointMassPotential(Potential):
    def __init__(self,M=1e3,zeropos=None):
        self.type = 'pointmass'
        self.M = M
        # Fix gauge
        if zeropos is None:
            self.gaugeparam = 0.0
        else:
            r0 = np.sqrt(zeropos[0]**2 + zeropos[1]**2)
            self.gaugeparam = G_grav*M/r0
    def phi(self,y):
        return -G_grav*self.M/np.sqrt(y[0]**2 + y[1]**2) + self.gaugeparam
    
    def accel(self,y):
        ff = -G_grav*self.M/((y[0]**2 + y[1]**2)**1.5)
        return np.array([y[0]*ff,y[1]*ff])

    def maxval_x(self,E):
        return G_grav*self.M/E

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