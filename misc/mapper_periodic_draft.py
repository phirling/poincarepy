from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import argparse
import solver
import scipy.optimize as scpopt
from potentials import *
import copy

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
        #self._evt = lambda t,y: crossing_function(t,y)
        #self._createevent(crossing_function)
        self._evt = copy.deepcopy(crossing_function)
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
        ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
        if ED < 0:
            return None
        else:
            y0 = [q[0],0.,q[1],np.sqrt(ED)]
            res = solver.integrate_orbit(self.pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1)
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
        deltq = q0
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
    def integrate_orbit(self,q,E,N=1,t_eval=None):
        if t_eval is None:
            t_eval = np.linspace(0,self.maxtime,10000)
        ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
        if ED < 0:
            return None
        else:
            y0 = [q[0],0.,q[1],np.sqrt(ED)]
            res = solver.integrate_orbit(self.pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1,t_eval=t_eval)
            return res['y_events'][0][:,[0,2]].T, res['y'][0:2]
    def xlim(self,E,x0=(-1,1)):
        def sp(x):
            A = np.zeros((4,x.shape[0]))
            A[0] = x
            return A
        g = lambda x: E-self.pot.phi(sp(x)[0:2])
        gprime = lambda x: self.pot.accel(sp(x))[0]
        root,conv,i = scpopt.newton(g,x0,gprime,full_output=True,maxiter=2000)
        if not conv.any():
            print("The root finding algorithm did not converge to find appropriate lims on axis {:n}".format(i))
            return None
        else:
            return 0.99999*root
    def vxlim(self,E,x):
        ED = 2*(E-self.pot.phi([x,0.]))
        return np.sqrt(ED)
        """if ED >= 0:
            return np.sqrt(ED)
        elif -1e-2 < ED < 0:
            return 0.0
        else:
            raise ValueError("x is out of range")"""

    def section(self,E,N_orbits=10,N_points=10,x0=(-1,1),t_eval = None):
        xl = self.xlim(E,x0)
        xx = np.linspace(xl[0],xl[1],N_orbits)
        xxx = np.linspace(xl[0],xl[1],400) # TODO: find a way with less points
        vx = self.vxlim(E,xxx)
        zvc = np.array([np.hstack((xxx,xxx[::-1])),np.hstack((vx,-vx[::-1]))])
        secs = []
        orbs = []
        for j,x in enumerate(xx):
            s,o = self.integrate_orbit([x,0],E,N_points,t_eval)
            secs.append(s)
            orbs.append(o)
        return secs, orbs, zvc
    """
    def lim(self,E,i=0,x0=(-1,1)):
        def sp(x,i):
            A = np.zeros((4,x.shape[0]))
            A[i] = x
            return A
        g = lambda x: E-self.pot.phi(np.array([x,np.zeros_like(x)]))
        gprime = lambda x: self.pot.accel(np.array([x,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x)]))[i]
        root,conv,i = scpopt.newton(g,x0,gprime,full_output=True,maxiter=2000)
        if not conv.any():
            print("The root finding algorithm did not converge to find appropriate lims on the x axis")
            return None
        else:
            return root
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-E",type=float,default=200)
    parser.add_argument("-q0",type=float,nargs=2,default=[0,0])
    parser.add_argument("-N_orbits",type=int,default=10)
    parser.add_argument("-N_points",type=int,default=20)
    parser.add_argument("-per",type=int,default=1)
    args = parser.parse_args()

    #pot = LogarithmicPotential(zeropos=(0,0))
    r0 = (0,0)
    logpot = LogarithmicPotential(zeropos=r0)
    rotpot = zRotation(0.3,zeropos=r0)

    pot = CombinedPotential(logpot,rotpot)
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

    fs = (15,7)
    fig, ax = plt.subplots(1,2,figsize=fs)
    #l1, = ax[0].plot([],[],'o')
    #l2, = ax[1].plot([],[])
    #ax[0].set_xlim(-8,8)
    #ax[0].set_ylim(-15,15)
    ax[1].axis('equal')
    

    secs,orbs,zvc = mapper.section(args.E,N_orbits=args.N_orbits,N_points=args.N_points)
    for s in secs:
        ax[0].plot(s[0],s[1],'o',ms=0.3,color='black')
    ax[0].plot(zvc[0],zvc[1])
    #print(mapper.xlim(args.E,x0=(-1,1)))
    #print(zvc)

    lper, = ax[0].plot([],[],'*',ms=8,color='green')
    lper_sec, = ax[0].plot([],[],'o',ms=7,color='green')
    lper_orb, = ax[1].plot([],[])
    def pick_y0(event):
        x, xdot = event.xdata, event.ydata
        q0 = [x,xdot]
        """
        sec,orb = mapper.integrate_orbit(q0,args.E,N=20)
        print(sec.shape)
        l1.set_xdata(sec[0])
        l1.set_ydata(sec[1])
        l2.set_xdata(orb[0])
        l2.set_ydata(orb[1])
        ax[1].relim()
        ax[1].autoscale()
        fig.canvas.draw()
        """
        qper = mapper.find_periodic_orbit(q0,args.E,N=args.per,print_result=True,
                                        print_progress=True,eps=1e-3)
        lper.set_xdata(qper[0])
        lper.set_ydata(qper[1])
        if 1:
            s,o = mapper.integrate_orbit(qper,args.E,N=20)
            lper_sec.set_xdata(s[0])
            lper_sec.set_ydata(s[1])
            lper_orb.set_xdata(o[0])
            lper_orb.set_ydata(o[1])
            ax[1].relim()
            ax[1].autoscale()
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', pick_y0)
    plt.show()
    #q0 = np.array(args.q0)
    #mapper.find_periodic_orbit(q0,args.E,print_result=True,print_progress=True)