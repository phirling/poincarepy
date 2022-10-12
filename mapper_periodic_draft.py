import numpy as np
import matplotlib.pyplot as plt
import argparse
import solver
import scipy.optimize as scpopt
from potentials import *

parser = argparse.ArgumentParser()

parser.add_argument("-E",type=float,default=40)
parser.add_argument("-q0",type=float,nargs=2,default=[0,0])
args = parser.parse_args()

def poincare_map(q,pot,E,max_time,crossfunction,nb_cross = 1):
    ED = 2*(E-pot.phi([q[0],q[1]])) - q[1]**2
    if ED < 0:
        return None
    else:
        y0 = [q[0],0.,q[1],np.sqrt(ED)]
        res = solver.integrate_orbit(pot.RHS,(0.,max_time),y0,events=crossfunction,event_count_end=nb_cross+1)
        return res['y_events'][0][-1][[0,2]] # x,vx components of last event

class PoincareMapper:
    def __init__(self,pot: Potential,crossing_function: callable,
                 max_integ_time=100,dx=1e-3,dvx=1e-3) -> None:
        self.pot = pot
        self.maxtime = max_integ_time
        self._evt = crossing_function
        self._dx = dx
        self._dvx = dvx
    def map(self,q,E,N=1):
        """Map a point q to its Poincare map after N crossings (in 2D)
        Parameters
        ----------
        q : array-like size (2,)
            Starting point in phase space
        E : float
            Energy of the orbit
        N : int
            Number of crossings (order of the map)
        
        Returns
        -------
        q' : array (2,) or None
            Result of the mapping. Returns None if the starting point was outside
            of the zero-velocity curve of the potential at energy E
        """
        ED = 2*(E-self.pot.phi([q[0],q[1]])) - q[1]**2
        if ED < 0:
            return None
        else:
            y0 = [q[0],0.,q[1],np.sqrt(ED)]
            res = solver.integrate_orbit(pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1)
            return res['y_events'][0][-1][[0,2]]
    def jac(self,q,E,N=1):
        """2D-Jacobian matrix of the map() function
        Parameters
        ----------
        q : array-like size (2,)
            Starting point in phase space
        E : float
            Energy of the orbit
        N : int
            Number of crossings (order of the map)
        
        Returns
        -------
        J : array (2,2)
            Jacobian matrix of the mapping. Returns None if the starting point was
            outside of the zero-velocity curve of the potential at energy E
        """
        if self.map(q,E,N) is None: return None # Need to deal with finite diffs outside zero vel curve
        Txf = self.map([q[0]+self._dx,q[1]],E,N)
        Txb = self.map([q[0]-self._dx,q[1]],E,N)
        Tvxf = self.map([q[0],q[1]+self._dvx],E,N)
        Tvxb = self.map([q[0],q[1]-self._dvx],E,N)
        #print([Txf,Txb,Tvxf,Tvxb])
        J00 = (Txf[0] - Txb[0]) / (2*self._dx)
        J01 = (Tvxf[0] - Tvxb[0]) / (2*self._dvx)
        J10 = (Txf[1] - Txb[1]) / (2*self._dx)
        J11 = (Tvxf[1] - Tvxb[1]) / (2*self._dvx)
        jac_matrix = np.array([[J00,J01],[J10,J11]])
        return jac_matrix
    def find_periodic_orbit(self,q0,E,N=1,print_result=False,print_progress=False,
                            maxiter = 100, eps = 1e-5):
        """Starting from q0 at energy E, find an N-periodic orbit
        Parameters
        ----------
        q0 : array-like size (2,)
            Initial guess of the periodic orbit search
        E : float
            Energy of the orbit
        N : int
            Number of crossings (order of the map)
        print_result : bool
            Print the found periodic orbit
        print_progress : bool
            Print the steps performed during the search
        maxiter: int
            Maximum number of iterations for the orbit search
        eps: float
            Precision |q_n-q_n-1| < eps required to find an orbit
        Returns
        -------
        q* : array (2,) or None
            Phase-space location of the found periodic orbit. Returns None if
            either the starting point was outside of the zero-velocity curve
            of the potential at energy E or if the search reached its maximum
            number of allowed iterations.
        """
        if self.map(q0,E,N) is None: return None
        F = lambda q: self.map(q,E,N) - q
        dF = lambda q: self.jac(q,E,N) - np.identity(2)
        ii = 0
        qn = np.asarray(q0)
        deltq = q
        while np.linalg.norm(deltq) > eps:
            ii += 1
            deltq = scpopt.lsq_linear(dF(qn),-F(qn))['x']
            # Check if the new point lies outside zero-vel curve
            while self.map(qn + deltq,E,N) is None:
                deltq /= 4.
            qn += deltq
            if ii > maxiter:
                print("Maximum number of iterations reached")
                return None
            if print_progress:
                print("#{:n}:".format(ii),end=" ")
                print(qn)
        if print_result:
            print("Converged to a periodic orbit after {:n} iterations:".format(ii))
            print("[x,vx] = [{:.3e},{:.3e}]".format(qn[0],qn[1]))
        return qn

if __name__ == "__main__":
    pot = LogarithmicPotential(zeropos=(0,0))
    def event_yplanecross(t,y):
        return y[1]
    event_yplanecross.direction = 1

    q = [-0.5,0]
    #max_time = 100
    #qprime = poincare_map(q,pot,args.E,max_time,event_yplanecross,nb_cross=1)
    #print(q)
    #print(qprime)
    mapper = PoincareMapper(pot,event_yplanecross)
    #print(mapper.map(q,args.E))
    #print(mapper.jac(q,args.E))
    q0 = np.array(args.q0)
    mapper.find_periodic_orbit(q0,args.E,print_result=True,print_progress=True)