import numpy as np
import matplotlib.pyplot as plt
import argparse
import solver
import scipy.optimize as scpopt
from potentials import *

parser = argparse.ArgumentParser()

parser.add_argument("-E",type=float,default=40)
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
        self._EPS = 1e-5
        self._maxiter = 100
    def map(self,q,E,N=1):
        ED = 2*(E-self.pot.phi([q[0],q[1]])) - q[1]**2
        if ED < 0:
            return None
        else:
            y0 = [q[0],0.,q[1],np.sqrt(ED)]
            res = solver.integrate_orbit(pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1)
            return res['y_events'][0][-1][[0,2]]
    def jac(self,q,E,N=1):
        if self.map(q,E,N) is None: return None
        Txf = self.map([q[0]+self._dx,q[1]],E,N)
        Txb = self.map([q[0]-self._dx,q[1]],E,N)
        Tvxf = self.map([q[0],q[1]+self._dvx],E,N)
        Tvxb = self.map([q[0],q[1]-self._dvx],E,N)
        J00 = (Txf[0] - Txb[0]) / (2*self._dx)
        J01 = (Tvxf[0] - Tvxb[0]) / (2*self._dvx)
        J10 = (Txf[1] - Txb[1]) / (2*self._dx)
        J11 = (Tvxf[1] - Tvxb[1]) / (2*self._dvx)
        jac_matrix = np.array([[J00,J01],[J10,J11]])
        return jac_matrix
    def find_periodic_orbit(self,q0,E,N=1):
        if self.map(q0,E,N) is None: return None
        F = lambda q: self.map(q,E,N) - q
        dF = lambda q: self.jac(q,E,N) - np.identity(2)
        ii = 0
        qn = q0
        deltq = q
        while np.linalg.norm(deltq) > self._EPS:
            ii += 1
            deltq = scpopt.lsq_linear(dF(qn),-F(qn))['x']
            qn += deltq
            if ii > self._maxiter:
                print("Maximum number of iterations reached")
                break
        print("Converged to a periodic orbit after " + str(ii) + " iterations:")
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
    print(mapper.map(q,args.E))
    print(mapper.jac(q,args.E))

    print(mapper.find_periodic_orbit(q,args.E))