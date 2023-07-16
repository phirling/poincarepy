from . import solver
import copy
import numpy as np
import scipy.optimize as scpopt
from tqdm import tqdm

# TODO: this can be done in a cleaner way
def event_yplanecross(t,y):
        return y[1]
event_yplanecross.direction = 1

class PoincareMapper:
    """Class that handles computations (orbit integration, map generation,..) based on a physical potential
    
    This is the base class that, provided with a Potential object, performs dynamics computation in
    this potential. These include: orbit integration, mapping, generation of Poincaré maps as well
    as collections of those at multiple energies, and periodic orbit searching.
    
    In the following, y is understood as a 4-vector containing the 4 dof of the system. By default, the map
    is drawn in the y[1] = 0 section of the phase space, and the y[3] variable is eliminated by the Hamiltonian
    integral. Thus, the map space is M = {y[0], y[2]}. In a cartesian frame, we would have y = [x,y,vx,vy] and
    M = {x,vx} with y=0 and vy calculated via H. In a meridonal plane frame, we would have y = [r,z,vr,vz] and
    M = {r,vr}, with z=0 and vz calculated via H.

    Other coordinate frames are not yet implemented and may lead to inconsistencies (if for example the kinetic
    energy is not in a cartesian form K = 0.5(y[2]^2 + y[3]^2)).

    Parameters
    ----------
    pot : Potential
        The physical potential that describes the dynamical system
    crossing_function : callable
        Condition function used to compute Poincaré maps, has the signature f(t,y) = 0 when
        a point is to be drawn. One sets the crossing direction by defining an attribute on the
        function object as f.direction = +1, -1, where +/- indicate the direction of crossing.
    max_integ_time: float
        The maximum allowed integration time (in physical/internal units). If a computation exceeds
        this limit, an error is raised. In general, the effective integration time in most computations
        will be determined by a condition (e.g. number of crossing events) rather than a total time,
        so this value only serves as an upper limit.
    dx: float
        Finite difference step used for the first dynamical variable (x,r,.) in the jacobian computation
    dvx: float
        Finite difference step used for the second dynamical variable (vx,vr,.) in the jacobian computation
    """
    def __init__(self,pot,crossing_function = event_yplanecross,
                 max_integ_time=200,dx=1e-8,dvx=1e-8) -> None:
        self.pot = pot
        self.maxtime = max_integ_time
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
        q' : array (2,)
            Result of the mapping. Returns None if the starting point was outside
            of the zero-velocity curve of the potential at energy E
        """
        ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
        if ED < 0:
            print("Cannot compute map (point outside ZVC)")
            return None
            #raise ValueError("Attempted to map a point outside ZVC")
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
        if not self._is_allowed(q,E):
            print("Cannot compute Jacobian (point outside ZVC)")
            return None
        Txf = self.map([q[0]+self._dx,q[1]],E,N)
        Txb = self.map([q[0]-self._dx,q[1]],E,N)
        Tvxf = self.map([q[0],q[1]+self._dvx],E,N)
        Tvxb = self.map([q[0],q[1]-self._dvx],E,N)
        #if Txf is None or Txb is None or Tvxf is None or Tvxb is None:
        if np.any([Txf,Txb,Tvxf,Tvxb] is None):
            print("Cannot compute Jacobian (neighbourhood too close to ZVC)")
            return None
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
        if not self._is_allowed(q0,E):
            print("Search starting point outside ZVC")
            return
        F = lambda q: self.map(q,E,N) - q
        dF = lambda q: self.jac(q,E,N) - np.identity(2)
        ii = 0
        qn = q0
        deltq = np.ones(2)
        while np.linalg.norm(deltq) > eps:
            ii += 1
            deltq = scpopt.lsq_linear(dF(qn),-F(qn))['x']
            while self.jac(qn + deltq,E,N) is None:
                deltq /= 2
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
    def integrate_orbit(self,q,E,N=1,N_points_orbit = 1000):
        """Integrate an orbit with given starting point

        Same principle as map() but the whole history of crossings, as well as the
        corresponding configuration-space orbit is returned, with a resolution
        provided as argument.

        Parameters
        ----------
        q : array-like size (2,)
            Starting point in phase space
        E : float
            Energy of the orbit
        N : int
            Number of crossings of the section plane
        N_points_orbit: int or None
            Number of points in the returned configuration space orbit. Increase for
            smoother orbits, decrease to reduce memory usage. Does not affect the computed
            sections in any way. Set to None to use integrator timesteps, which is faster, but
            cannot be used to generate sections since the number of points is then variable.

        Returns
        -------
        sec : array (2,N)
            Phase space section of the integrated orbit, i.e. the N points at which
            the orbit has crossed the section plane in a given direction
        orb: array(2,*)
            Configuration space orbit, length depending on integration time required
            to reach N crossings
        """
        ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
        if ED < 0:
            raise ValueError("Starting point out of range")
        else:
            y0 = [q[0],0.,q[1],np.sqrt(ED)]
            if N_points_orbit is not None:
                res = solver.integrate_orbit(self.pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1,dense_output=True)
                sol = res['sol']
                ts = np.linspace(0,res['t'][-1],N_points_orbit)
                orb = sol(ts)
                return res['y_events'][0][1:,[0,2]].T, orb[0:2] # Exclude first event
            else:
                res = solver.integrate_orbit(self.pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1,t_eval=None)
                return res['y_events'][0][1:,[0,2]].T, res['y'][0:2] # Exclude first event

    def integrate_orbit_full(self,y0,tf,N_points_orbit = 1000):
        """Integrate an orbit with given starting point in the full phase space

        Given a point y0 in the 4D phase space, integrate until tf. This method does not
        calculate crossing points and outputs only a phase space orbit.

        Parameters
        ----------
        y0 : 1D-array of size 4
            The starting point (x,y,vx,vy) in phase space
        tf : float
            Integration time
        N_points_orbit : int
            Number of samples in (0,tf) at which to store the solution. The sampling times
            are uniformly distributed in (0,tf)

        Returns
        -------
        yt : 2D-array of size (4,N_points_orbit)
            The trajectory (x(t),y(t),vx(t),vy(t)) in phase space
        """
        t_eval = np.linspace(0,tf,N_points_orbit)
        res = solver.integrate_orbit(self.pot.RHS,(0.,tf),y0,t_eval)
        return res['y']

    def section(self,E,xlim,N_orbits,N_points,xdot=0.,auto_lim=True,N_points_orbit = 10000,Nsteps_lim=20,print_progress=False):
        """Calculate a surface of section at given energy

        A surface of section is a collection of N_orbits orbits that fill a given region of
        the phase space. Each orbit is integrated until N_points crossings of the section
        plane occur, and at each (oriented) crossing, a point is added to the surface. Phase
        space is filled by considering ICs distributed linearly in a given or calculated
        x-range, with xdot_0 a constant value.

        Parameters
        ----------
        E : float
            Energy of the section
        xlim: array-like or tuple
            If auto_lim is True, lower and upper bounds used for the automatic xlim search. If
            auto_lim is False, these are used as-is to compute the section.
        N_orbits: int
            Number of orbits in the surface of section
        N_points: int
            Number of points per orbit in the map (e.g number of crossings of the section plane)
        xdot: float
            Constant scalar value for the xdot (y) initial condition
        auto_lim: bool
            Whether or not to automatically determine xlims of the section
        N_points_orbit: int or None
            Number of points per orbit in the configuration space. Increase for
            smoother orbits, decrease to reduce memory usage. If set to None, no orbit output
            will be produced and internally the integrator won't need to do any interpolation,
            which makes it run moderately faster.
        Nsteps_lim: int
            Number of subdivisions of the interval to use when auto-searching. Use higher value
            for larger intervals.

        Returns
        -------
        secs : array (N_orbits,2,N_points)
            The surface of section. First dim indexes the orbits, second the variable (x,xdot),
            third are the points
        orbs: list (N_orbits,) of arrays (2,*)
            Configuration-space orbits corresponding to the sections. List because the number of points
            in the orbit is not constant (deps on the integration time that was required to reach
            N_points crossings of the plane). If N_points_orbit = None, orbs = None
        zvc: array (2,800)
            Zero-velocity curve (limit of the section). Consists of (x,xdot) pairs that contour the
            surface of section once.
        """
        if auto_lim:
            xl = self.xlim(E,xdot,xlim[0],xlim[1],Nsteps_lim)
            # Add tolerance margin TODO: fix
            if xl[0] < 0: xl[0]*=0.999
            else: xl[0]*=1.001
            if xl[1] > 0: xl[1]*=0.999
            else: xl[1]*=1.001
        else:
            xl = np.asarray(xlim)
        #print(xl)
        if not self._is_allowed([xl,0,xdot,0],E):
            raise ValueError("The provided xlim/xdot lie outside the allowed ZVC")
        if print_progress:
            pp = lambda x: tqdm(x)
        else:
            pp = lambda x: x
        xx = np.linspace(xl[0],xl[1],N_orbits)
        xxx = np.linspace(xl[0],xl[1],400) # TODO: find a way with less points
        vx = self.zvc(E,xxx)
        zvc = np.array([np.hstack((xxx,xxx[::-1])),np.hstack((vx,-vx[::-1]))])
        if N_points_orbit is not None:
            secs = np.empty((N_orbits,2,N_points))
            orbs = np.empty((N_orbits,2,N_points_orbit))
            for j,x in enumerate(pp(xx)):
                s,o = self.integrate_orbit([x,xdot],E,N_points,N_points_orbit)
                secs[j] = s
                orbs[j] = o
            return secs, orbs, zvc
        else:
            # Special mode for redrawing when no orbit output is desired
            secs = np.empty((N_orbits,2,N_points))
            for j,x in enumerate(pp(xx)):
                s,o = self.integrate_orbit([x,xdot],E,N_points,N_points_orbit=None)
                secs[j] = s
            return secs, None, zvc

    def section_collection(self,E,xlim,N_orbits,N_points,xdot=0.,N_points_orbit = 10000,
                            auto_lim = True,Nsteps_lim = 200):
        """Calculate a set of sections at different energies

        Parameters
        ----------
        E : array
            Energies of the sections
        N_orbits: int
            Number of orbits in each surface of section
        N_points: int
            Number of points per orbit in the maps (e.g number of crossings of the section plane)
        xlim: array-like or tuple
            If auto_lim is True, lower and upper bounds used for the automatic xlim search. If
            auto_lim is False, these are used as-is to compute the section.
        xdot: array
            Constant scalar values for the xdot (y) initial condition, used if xlim is explicitely provided
        x0: array of tuples
            Starting point (initial guess) for the automatic xlim computation
        N_points_orbit: int
            Number of points for the orbit output. If None, the values at each time step
            of the integrator will be used (faster)
        auto_lim: bool
            Whether or not to automatically determine xlims of the section
            
        Returns
        -------
        sections : array (N_energies,N_orbits,2,N_points)
            The surface of section. First dim indexes the orbits, second the variable (x,xdot),
            third are the points
        orbits: array (N_energies,N_orbits,2,N_points_orbit)
            Configuration-space orbits corresponding to the sections. List because the number of points
            in the orbit is not constant (deps on the integration time that was required to reach
            N_points crossings of the plane)
        zvcs: array (N_energies,2,800)
            Zero-velocity curves (limits of the sections). Each entry consists of (x,xdot) pairs that
            contour that surface of section once.
        
        """
        N_E = E.shape[0]
        orbits = np.empty((N_E,N_orbits,2,N_points_orbit))
        sections = np.empty((N_E,N_orbits,2,N_points))
        zvcs = np.empty((N_E,2,800))
        for j,e in enumerate(tqdm(E)):
            #print("Map #{:n} at E = {:.2f}".format(j+1,e))
            s,o,zvc = self.section(e,xlim,N_orbits,N_points,xdot,auto_lim,N_points_orbit,Nsteps_lim)
            orbits[j] = o
            sections[j] = s
            zvcs[j] = zvc
        return sections,orbits,zvcs

    def xlim(self,E,xdot,a,b,Nsteps=200):
        """Helper function to calculate physical x-limits at given energy

        Use the condition E=phi with y=0, ydot=0 to compute the maximum
        allowed values for x at given xdot.

        Parameters
        ----------
        E: float
            Energy of the map
        xdot: float
            Conjugate variable to x
        a: float
            Lower bound for the search
        b: float
            Upper bound for the search
        Nsteps: int
            Number of subdivisions of [a,b] to use to find the roots. Default is 200
            (Use larger value if a large interval is provided)
        """
        xs = np.linspace(a,b,Nsteps)
        phidiff = self.pot.phi([xs,np.zeros_like(xs)]) - E - 0.5*xdot**2
        f = lambda x: self.pot.phi([x,np.zeros_like(x)]) - E - 0.5*xdot**2
        zeros = []
        for i in range(Nsteps-1):
            if phidiff[i]*phidiff[i+1] > 0:
                continue
            else:
                root = scpopt.brentq(f,xs[i],xs[i+1])
                zeros.append(root)
        zeros = np.asarray(zeros)
        nzeros = zeros.shape[0]
        if nzeros == 2:
            return zeros
        elif nzeros  == 0:
            raise RuntimeError("No roots found in E-phi. Use larger Nsteps or tighten xlim")
        elif nzeros%2 == 0:
            mid = int(nzeros / 2)
            return zeros[mid-1:mid+1]
        else:
            raise RuntimeError("Non-integer number of roots found in E-phi")

        
    def zvc(self,E,x):
        """Helper function to calculate zero velocity curve
        """
        ED = 2*(E-self.pot.phi([x,0.]))
        return np.sqrt(ED)

    def _is_allowed(self,q,E):
        ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
        if np.any(ED < 0): return False
        else: return True