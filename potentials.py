import numpy as np


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
        return [y[2],
                y[3],
                F[0],
                F[1]]
    def info(self):
        s = "Empty Potential"
        return(s)


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