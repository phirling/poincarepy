from poincarepy import solver
from poincarepy import potentials as pot

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button, TextBox, RectangleSelector
import copy
import numpy as np
import scipy.optimize as scpopt
from tqdm import tqdm

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
    def __init__(self,pot: pot.Potential,crossing_function = event_yplanecross,
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
    def integrate_orbit(self,q,E,N=1,N_pts_orbit = 1000,dense=True):
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
        N_pts_orbit: int
            Maximum number of points in the returned config. space orbit array. Max
            because in reality integration stops when N crossings have occured, so the
            length of the output orbit array is N'<= N_pts_orbit

        Returns
        -------
        sec : array (2,N)
            Phase space section of the integrated orbit, i.e. the N points at which
            the orbit has crossed the section plane in a given direction
        orb: array(2,*)
            Configuration space orbit, length depending on integration time required
            to reach N crossings
        """
        # This is a depcrecated mode, kept only for documentation but will be deleted
        if not dense:
            if N_pts_orbit is not None:
                t_eval = np.linspace(0,self.maxtime,10000)
            else:
                t_eval = None
            ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
            if ED < 0:
                raise ValueError("Starting point out of range")
            else:
                y0 = [q[0],0.,q[1],np.sqrt(ED)]
                res = solver.integrate_orbit(self.pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1,t_eval=t_eval)
                return res['y_events'][0][1:,[0,2]].T, res['y'][0:2] # Exclude first event
        else:
            ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
            if ED < 0:
                raise ValueError("Starting point out of range")
            else:
                y0 = [q[0],0.,q[1],np.sqrt(ED)]
                res = solver.integrate_orbit(self.pot.RHS,(0.,self.maxtime),y0,events=self._evt,event_count_end=N+1,dense_output=True)
                sol = res['sol']
                ts = np.linspace(0,res['t'][-1],N_pts_orbit)
                orb = sol(ts)
                return res['y_events'][0][1:,[0,2]].T, orb[0:2] # Exclude first event


    def section(self,E,xlim,N_orbits,N_points,xdot=0.,auto_lim=True,nb_pts_orbit = 10000,Nsteps_lim=20,print_progress=False):
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
        nb_pts_orbit: int
            (Maximum) number of points for the orbit output. If None, the values at each time step
            of the integrator will be used (faster)
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
            N_points crossings of the plane)
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
        if not self._is_allowed([xl,0,xdot,0],E):
            raise ValueError("The provided xlim/xdot lie outside the allowed ZVC")

        xx = np.linspace(xl[0],xl[1],N_orbits)
        xxx = np.linspace(xl[0],xl[1],400) # TODO: find a way with less points
        vx = self.zvc(E,xxx)
        zvc = np.array([np.hstack((xxx,xxx[::-1])),np.hstack((vx,-vx[::-1]))])
        secs = np.empty((N_orbits,2,N_points))
        orbs = []
        if print_progress:
            pp = lambda x: tqdm(x)
        else:
            pp = lambda x: x
        for j,x in enumerate(pp(xx)):
            s,o = self.integrate_orbit([x,xdot],E,N_points,nb_pts_orbit)
            secs[j] = s
            orbs.append(o)
        return secs, orbs, zvc
    def section_collection(self,E,xlim,N_orbits,N_points,xdot=0.,nb_pts_orbit = 10000,
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
        xlim: array of tuples
            x-limits of the initial conditions. If none are provided, the maximum physically allowed
            values are computed from E=phi
        xdot: array
            Constant scalar values for the xdot (y) initial condition, used if xlim is explicitely provided
        x0: array of tuples
            Starting point (initial guess) for the automatic xlim computation
        nb_pts_orbit: int
            (Maximum) number of points for the orbit output. If None, the values at each time step
            of the integrator will be used (faster)
            
        Returns
        -------
        sections : array (N_energies,N_orbits,2,N_points)
            The surface of section. First dim indexes the orbits, second the variable (x,xdot),
            third are the points
        orbits: list (N_energies,N_orbits) of arrays (2,*)
            Configuration-space orbits corresponding to the sections. List because the number of points
            in the orbit is not constant (deps on the integration time that was required to reach
            N_points crossings of the plane)
        zvcs: array (N_energies,2,800)
            Zero-velocity curves (limits of the sections). Each entry consists of (x,xdot) pairs that
            contour that surface of section once.
        
        """
        N_E = E.shape[0]
        orbits = []
        sections = np.empty((N_E,N_orbits,2,N_points))
        zvcs = np.empty((N_E,2,800))
        for j,e in enumerate(tqdm(E)):
            #print("Map #{:n} at E = {:.2f}".format(j+1,e))
            s,o,zvc = self.section(e,xlim,N_orbits,N_points,xdot,auto_lim,nb_pts_orbit,Nsteps_lim)
            orbits.append(o)
            sections[j] = s
            zvcs[j] = zvc
        return sections,orbits,zvcs

    def xlim(self,E,xdot,a,b,Nsteps=30):
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
            Number of subdivisions of [a,b] to use to find the roots. Default is 30
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
            if len(zeros) == 2:
                break
        if len(zeros) == 0:
            raise RuntimeError("No roots found! Use larger Nsteps")
        else:
            return np.asarray(zeros)
        
    def zvc(self,E,x):
        """Helper function to calculate zero velocity curve
        """
        ED = 2*(E-self.pot.phi([x,0.]))
        return np.sqrt(ED)

    def _is_allowed(self,q,E):
        ED = 2*(E-self.pot.phi([q[0],0.])) - q[1]**2
        if np.any(ED < 0): return False
        else: return True

class PoincareCollection:
    """Container class for a collection of surfaces of section and related data, used for pickling

    This class is essentially a dictionnary of the elements making up a Poincaré Map collection, that
    are required by the Tomography class. It is used only for convenience when saving (pickling)
    calculated data for reuse.

    Parameters
    ----------
    E_list: list or array
        Array of energy values corresponding to the energies of each set of orbits
    orbits_list: list
        List of shape (N_energies,N_orbits) where each element is again a list corresponding to
        a set of orbits at a given energy. An orbit is an array of shape (2,N) with the first
        axis giving x/y and N the number of points in the orbit.
    sections_list: list
        Same idea as for the orbits, except that a surface of section is an array of shape
        (2,nb_points) with nb_points the number of points in the Poincare map per orbit
        (number of crossings of the x/y plane)
    potential: Potential
        Potential that generated the collection (used if the collection is imported
        and exported)

    Attributes
    ----------
    energylist: list
        E_list
    orbitslist: list
        orbits_list
    sectionslist: list
        sections_list
    nb_energies: int
        Number of energy levels in the collection (= len(energylist))
    nb_orbits_per_E: int
        Number of orbits per energy
    
    (The total number of orbits in a collection is nb_energies x nb_orbits_per_E)
    """
    
    def __init__(self,E_list,orbits_list,sections_list: np.ndarray,zvc_list: np.ndarray,mapper: PoincareMapper) -> None:
        if not (len(E_list) == len(orbits_list) == sections_list.shape[0]):
            raise ValueError("lists must be of the same length")
        self.energylist = E_list
        self.orbitslist = orbits_list
        self.sectionsarray = sections_list
        self.zvc_list = zvc_list
        self.mapper = mapper
        self.nb_energies = len(E_list)
        self.nb_orbits_per_E = len(orbits_list[0])


class Tomography:
    """Tomographic visualisation of an ensemble of surfaces of section at different energies

    Main visualization class. Using precalculated data (Poincaré sections at different energies), allows
    the user to pan through the energy dimension in a tomography-like fashion and see how the phase space
    of the system (represented with a Poincaré section) changes in this dimension. One can click on a given
    orbit in the reduced phase space to see how it looks like in configuration space.

    In addition, if the user wishes to analyze some part of the phase space in more detail, he or she can
    redraw orbits in real time, using one of 3 functions:
    - Single orbit redrawing: This mode is toggled using the "z" key and, by clicking on a point in the reduced
    phase space, a new orbit is calculated using the initial conditions derived from the clicked point.
    - Rectangular selection redrawing: This mode is toggled using the "t" key and allows the user to select a
    rectangular region in which a desired N number of orbits, with ICs lying uniformly on the middle horizontal
    of the rectangle, are redrawn.
    - Full view redraw: Is toggled with a button and simply redraws the section at the current energy level with
    N orbits, where N can be set in a text box.

    The class methods are essentially all internal and are thus not described in detail in the API.

    Parameters
    ----------
    sections: 4D-array (N_E,N_orbits,2,N_points)
        Main data array, contains the points of the surfaces of section at every energy. First dim
        indexes the energy, second the number of the orbit in the section, third the coordinate, fourth are
        the data.
    orbits: list (N_E,) of lists (N_orbits,) of arrays (2,*)
        Configuration space orbits corresponding to the sections. Same layout but with lists, since
        the data length of the individual orbits varies due to different integration times
    zvcs: 3D-array (N_E,2,800)
        Zero-velocity curves of each surface of section
    energies: array (N_E,)
        Energies of the surfaces of section
    mapper: PoincareMapper
        PoincareMapper instance used to generate the data, required for the redrawing & orbit search
        features.
    figsize: tuple of floats, optional
        Size of the figure in inches
    redraw_orbit: bool, optional
        Whether to redraw a selected orbit when switching to a different energy. Default is true
    """
    
    """
    Figure layout:
    Horizontal: <0.03> margin <0.1> buttons <0.05> margin <0.36> axes <0.07> margin <0.36> axes <0.03> margin
    Vertical: <0.08> margin <*> axes
    """
    def __init__(self,sections: np.ndarray ,orbits, zvcs: np.ndarray,energies, mapper: PoincareMapper,
                 figsize=(12.5,6),title=None, redraw_orbit: bool = True,
                 axlabels=["$x$","$\dot{x}$","$x$","$y$"]) -> None:
        """ Load Data """
        self._sl = sections
        self._ol = orbits
        self._zvcl = zvcs
        self._El = energies
        self.mapper = mapper
        
        self._nEn = len(sections)
        self._nSec = len(sections[0])

        # Precompute figure limits per energy
        x0, x1 = np.amin(zvcs[:,0],axis=1),np.amax(zvcs[:,0],axis=1)
        y0, y1 = np.amin(zvcs[:,1],axis=1),np.amax(zvcs[:,1],axis=1)
        self.axlims = np.array([
            x0 - 0.05*(x1-x0),
            x1 + 0.05*(x1-x0),
            y0 - 0.05*(y1-y0),
            y1 + 0.05*(y1-y0)
        ])

        ## Construct figure
        # Main fig & axes
        self.fig = plt.figure(figsize=figsize)
        aspct = figsize[0] / figsize[1]
        axw = 0.33
        axstart = 0.1
        self.ax_sec = self.fig.add_axes([0.23,0.1,axw,axw * aspct])
        self.ax_orb = self.fig.add_axes([0.64,0.1,axw,axw * aspct])
        self.ax_pot = self.fig.add_axes([0.035,0.56,0.15,0.365])
        self.ax_orb.axis('equal')
        ffs = 14
        self.ax_sec.set_xlabel(axlabels[0],fontsize=ffs)
        self.ax_sec.set_ylabel(axlabels[1],fontsize=ffs)
        self.ax_orb.set_xlabel(axlabels[2],fontsize=ffs)
        self.ax_orb.set_ylabel(axlabels[3],fontsize=ffs)
        self.ax_sec.set_title("Section Plane",fontsize=ffs)
        self.ax_orb.set_title("Orbital Plane",fontsize=ffs)
        self.ax_pot.set_xlabel(axlabels[0])
        self.ax_pot.set_title("$\phi$")
        if title is not None:
            self.fig.text(0.6,axw * aspct + 0.2,title,fontsize=ffs+5,ha='center')

        self.lines_sec = [self.ax_sec.plot([], [],'o',ms=0.3,color='black',picker=True,pickradius=5)[0] for i in range(self._nSec)]
        self.line_orb = self.ax_orb.plot([], [],lw=1,color='tomato')[0]
        self.line_zvc = self.ax_sec.plot([], [],lw=0.5,color='indianred')[0]

        # Potential
        xpot = np.linspace(x0,x1,100)
        phi = self.mapper.pot.phi([xpot,0])
        self.ax_pot.plot(xpot,phi,color='black')
        self.Eline = self.ax_pot.axhline(self._El[0],color='indianred')
        self.Etext = self.ax_pot.text(0.75, 0.9,'',color='indianred',transform=plt.gca().transAxes)

        # Quit button
        ax_quitbutton = self.fig.add_axes([0.035, 0.05, 0.12, 0.075])
        self.button_quit = Button(ax_quitbutton,"Quit",color='mistyrose',hovercolor='lightcoral')
        self.button_quit.on_clicked(self._quitfig)
        
        # Set orbit search period with a Text box
        ax_setperiod = self.fig.add_axes([0.095, 0.325, 0.06, 0.05])
        self.textbox_setperiod = TextBox(ax_setperiod,'$p=$  ',color='mistyrose',
                    hovercolor='lightcoral', initial=1)
        self.textbox_setperiod.on_submit(self._set_search_period)
        self._p = 1

        # Button to toggle search mode
        ax_searchbutton = self.fig.add_axes([0.035, 0.4, 0.12, 0.075])
        self.button_search = Button(ax_searchbutton,'Search for\np-periodic orbits',
                    color='mistyrose',hovercolor='lightcoral')
        self.button_search.on_clicked(self._toggle_searchmode)
        self.line_psection, = self.ax_sec.plot([],[],'o',ms=4,color='mediumspringgreen')
        self._in_searchmode = False

        # Main Interactivity (switch energy, click on section)
        self._firstpick = True
        self.redraw_orbit = redraw_orbit
        self.idx = 0
        self.fig.canvas.mpl_connect('key_press_event',self)
        self._pickid = self.fig.canvas.mpl_connect('pick_event',self._onpick)
        self._in_redrawmode = False

        # Redrawing Functions
        ax_setredraw_N = self.fig.add_axes([0.095, 0.15, 0.06, 0.05])
        self.textbox_setredraw = TextBox(ax_setredraw_N,'$N_{{Redraw}}=$  ',color='mistyrose',
                    hovercolor='lightcoral', initial=10)
        self._Nredraw = 10
        self.textbox_setredraw.on_submit(self._set_redraw_N)

        # Redraw current view
        ax_redrawbutton = self.fig.add_axes([0.035, 0.225, 0.12, 0.075])
        self.button_redraw = Button(ax_redrawbutton,'Redraw current\nview',
                    color='mistyrose',hovercolor='lightcoral')
        self.button_redraw.on_clicked(self._redrawcurrent)

        # Redraw rectangle selection
        self._selector = RectangleSelector(self.ax_sec,self._selectandredraw)
        self._selector.set_active(False)
        self.fig.canvas.mpl_connect('key_press_event',self._toggle_rectsel)

        # Redraw single orbit
        self._singleredrawmode = False
        self.fig.canvas.mpl_connect('key_press_event',self._toggle_singleredraw)

        # Info
        #infotxt = self.fig.text(0.03,0.5,"Navigation:\nup/down: Switch Energy level\nt: Rectangle select & redraw\nz: Single orbit redraw")
        # Show lowest energy to start
        self.show(0)
        plt.show()
    def __call__(self,event):
        """Interaction function to switch energy level by up/down keys
        Parameters
        ----------
        event : matplotlib.key_press_event
            Up or down key press event
        """
        up = event.key == 'up'
        down = event.key == 'down'
        ii = self.idx
        if up:
            ii += 1
        elif down:      
            ii -= 1
        if up or down:
            if self._in_redrawmode:
                self._exit_redraw()
            if ii in range(self._nEn):
                self.line_psection.set_xdata([])
                self.line_psection.set_ydata([])
                self.show(ii)
                self.idx = ii
    def show(self,idx):
        """Function that updates the plot when energy is changed
        Parameters
        ----------
        idx : int
            Index of the energy level to switch to
        """

        for k,l in enumerate(self.lines_sec):
            l.set_xdata(self._sl[idx,k,0])
            l.set_ydata(self._sl[idx,k,1])
        self.line_zvc.set_xdata(self._zvcl[idx,0])
        self.line_zvc.set_ydata(self._zvcl[idx,1])
        # Set ax limits to 1.05*zero velocity curve
        mgn = 1.05
        #self.ax_sec.set_xlim(mgn*self._zvcl[idx,0,0],mgn*self._zvcl[idx,0,399])
        #self.ax_sec.set_ylim(mgn*self._zvcl[idx,1,599],mgn*self._zvcl[idx,1,199])
        self.ax_sec.set_xlim(self.axlims[0,idx],self.axlims[1,idx])
        self.ax_sec.set_ylim(self.axlims[2,idx],self.axlims[3,idx])
        if not self._firstpick and self.redraw_orbit:
            self.line_orb.set_xdata(self._ol[idx][self.artistid][0])
            self.line_orb.set_ydata(self._ol[idx][self.artistid][1])
            self.ax_orb.relim()
            self.ax_orb.autoscale()
        self.Eline.set_ydata(self._El[idx])
        self.Etext.set_text('{:.2f}'.format(self._El[idx]))
        self.fig.canvas.draw()
    def _onpick(self,event):
        """Interaction function to show an orbit by picking a surface of section
        Parameters
        ----------
        event : matplotlib.pick_event
            Event of picking a surface of section
        """
        if self._firstpick:
            self._firstpick = False
        else:
            self.prev_artist.set_color('black')
            self.prev_artist.set_markersize(0.3)
        event.artist.set_color('orangered')
        event.artist.set_markersize(1.5)
        self.artistid = self.lines_sec.index(event.artist)
        self._set_orb(self._ol[self.idx][self.artistid])
        self.fig.canvas.draw()
        self.prev_artist = event.artist

    def _quitfig(self,event):
        print("Program was exited by the user")
        plt.close(self.fig)
    
    # Redrawing helper functions
    def _redraw(self,x,y,orbit=None,hide_other=True,s=0.3,c='black'):
        if self._in_redrawmode:
            self._clear_redraw()
        else:
            self._in_redrawmode = True
            if hide_other:
                for l in self.lines_sec:
                    l.set_visible(False)
        self.lstmp = self.ax_sec.scatter(x, y,s=s,c=c)
        if orbit is not None: self._set_orb(orbit)
        self.fig.canvas.draw()
    def _clear_redraw(self):
        if hasattr(self,"lstmp"):
            self.lstmp.remove()
    def _exit_redraw(self):
        self._clear_redraw()
        delattr(self,"lstmp")
        self._in_redrawmode = False
        for l in self.lines_sec:
            l.set_visible(True)
    def _set_redraw_N(self,N):
        self._Nredraw = int(N)

    # Redraw Current view (whole fig or zoom)
    def _redrawcurrent(self,event):
        # Set xlim to current zoom
        xl_phys = (np.amin(self._zvcl[self.idx,0]),np.amax(self._zvcl[self.idx,0]))
        xl_fig = self.ax_sec.get_xlim()
        vxl = self.ax_sec.get_ylim()
        xl = (max(xl_fig[0],xl_phys[0]),min(xl_fig[1],xl_phys[1]))
        xdot0 = (vxl[0] + vxl[1])/2.
        print("Redrawing {:n} orbits...".format(self._Nredraw))
        stmp,otmp,z = self.mapper.section(self._El[self.idx],
                    N_orbits=self._Nredraw,N_points=len(self._sl[0][0][0]),
                    nb_pts_orbit=None,xlim=xl,xdot=xdot0,
                    print_progress=True,auto_lim=False)
        x = stmp[:,0,:].flatten()
        y = stmp[:,1,:].flatten()
        self._redraw(x,y)

    # Redraw rectangle selection
    def _toggle_rectsel(self,event):
        if event.key == 't':
            if self._selector.get_active() == False:
                print("Entering select & redraw mode")
                self._selector.set_active(True)
            else:
                print("Leaving select & redraw mode")
                self._selector.set_active(False)
    def _selectandredraw(self,eclick,erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xdot0 = (y1+y2)/2.
        stmp,otmp,z = self.mapper.section(self._El[self.idx],xlim=(x1,x2),N_orbits=self._Nredraw,N_points=len(self._sl[0][0][0]),nb_pts_orbit=None,auto_lim=False,xdot=xdot0)
        x = stmp[:,0,:].flatten()
        y = stmp[:,1,:].flatten()
        self._redraw(x,y)
    
    # Draw single additional orbit
    def _toggle_singleredraw(self,event):
        if event.key == 'z':
            if self._singleredrawmode == False:
                print("Entering single orbit redraw mode")
                self._singleredrawmode = True
                self.fig.canvas.mpl_disconnect(self._pickid)
                self._singleredrawid = self.fig.canvas.mpl_connect('button_press_event',self._singleredraw)
            else:
                print("Leaving single orbit redraw mode")
                self._singleredrawmode = False
                self._pickid = self.fig.canvas.mpl_connect('pick_event',self._onpick)
                self.fig.canvas.mpl_disconnect(self._singleredrawid)
                self._exit_redraw()
    def _singleredraw(self,event):
        if event.inaxes == self.ax_sec:
            q0 = [event.xdata,event.ydata]
            s,o = self.mapper.integrate_orbit(q0,self._El[self.idx],len(self._sl[0][0][0]))
            self._redraw(s[0],s[1],o,hide_other=False,s=5,c='rebeccapurple')

    # Periodic Orbit search
    def _set_search_period(self,p):
        self._p = int(p)
    def _toggle_searchmode(self,event):
        if not self._in_searchmode:
            self.button_search.color = 'firebrick'
            self._in_searchmode = True
            self._clickid = self.fig.canvas.mpl_connect('button_press_event',self._search)
            self.line_orb.set_xdata([])
            self.line_orb.set_ydata([])
            self.line_psection.set_visible(True)
            self.fig.canvas.mpl_disconnect(self._pickid)
        else:
            self.button_search.color = 'mistyrose'
            self._in_searchmode = False
            self.fig.canvas.mpl_disconnect(self._clickid)
            self.line_psection.set_xdata([])
            self.line_psection.set_ydata([])
            self.line_orb.set_xdata([])
            self.line_orb.set_ydata([])
            self.line_psection.set_visible(False)
            self._pickid = self.fig.canvas.mpl_connect('pick_event',self._onpick)
    def _search(self,event):
        if event.inaxes == self.ax_sec:
            E = self._El[self.idx]
            q0 = [event.xdata,event.ydata]           
            qstar = self.mapper.find_periodic_orbit(q0,E,
                    self._p,print_progress=True,eps=1e-3,
                    maxiter=100,print_result=True)
            if qstar is not None:
                if 0:
                    # TODO: stability
                    eigvals = np.linalg.eigvals(self.mapper.jac(qstar,E,self._p))
                    print(eigvals)
                    print(np.abs(eigvals))
                s,o = self.mapper.integrate_orbit(qstar,E,N=5*self._p)
                self.line_psection.set_xdata(s[0])
                self.line_psection.set_ydata(s[1])
                self._set_orb(o)
                self.fig.canvas.draw()
            else:
                print("The orbit finder did not converge with the provided starting guess")
    def _set_orb(self,o):
        self.line_orb.set_xdata(o[0])
        self.line_orb.set_ydata(o[1])
        self.line_orb.axes.relim()
        self.line_orb.axes.autoscale()