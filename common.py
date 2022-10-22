import solver
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button, TextBox
from potentials import Potential
import copy
import numpy as np
import scipy.optimize as scpopt

def event_yplanecross(t,y):
        return y[1]
event_yplanecross.direction = 1

class PoincareMapper:
    def __init__(self,pot: Potential,crossing_function = event_yplanecross,
                 max_integ_time=200,dx=1e-3,dvx=1e-3) -> None:
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
    def integrate_orbit(self,q,E,N=1,N_pts_orbit = 10000):
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
    def section(self,E,N_orbits=10,N_points=10,xlim=None,xdot0=None,x0=(-1,1),nb_pts_orbit = 10000):
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
        N_orbits: int
            Number of orbits in the surface of section
        N_points: int
            Number of points per orbit in the map (e.g number of crossings of the section plane)
        xlim: tuple of floats
            x-limits of the initial conditions. If none are provided, the maximum physically allowed
            values are computed from E=phi
        xdot0: float
            Constant scalar value for the xdot (y) initial condition
        x0: tuple of floats
            Starting point (initial guess) for the automatic xlim computation
        nb_pts_orbit: int
            (Maximum) number of points for the orbit output. If None, the values at each time step
            of the integrator will be used (faster)
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
        if xlim is None:
            xl = self.xlim(E,x0)
        else:
            xl = xlim
        if xdot0 is None:
            xdot = 0.
        else:
            xdot = xdot0
        xx = np.linspace(xl[0],xl[1],N_orbits)
        xxx = np.linspace(xl[0],xl[1],400) # TODO: find a way with less points
        vx = self.vxlim(E,xxx)
        zvc = np.array([np.hstack((xxx,xxx[::-1])),np.hstack((vx,-vx[::-1]))])
        secs = np.empty((N_orbits,2,N_points))
        orbs = []
        for j,x in enumerate(xx):
            s,o = self.integrate_orbit([x,xdot],E,N_points,nb_pts_orbit)
            secs[j] = s
            orbs.append(o)
        return secs, orbs, zvc
    def xlim(self,E,x0=(-1,1)):
        """Helper function to calculate physical x-limits at given energy

        Use the condition E=phi with y=0, xdot=0 to compute the maximum
        allowed values for x.
        """
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
        """Helper function to calculate xdot from x and E, with y=0
        """
        ED = 2*(E-self.pot.phi([x,0.]))
        return np.sqrt(ED)

    

class PoincareCollection:
    """Container class for a collection of orbits/Poincare maps at multiple energies

    This class is used to store the data corresponding to a collection of integrated orbits.
    A collection is a list consisting of multiple sets of orbits, where one
    set is at a given energy. The class also contains an array indicating the energies
    corresponding to each set.

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
    def get_potential(self):
        return self.potential


class Tomography:
    """Tomographic visualisation of a PoincareCollection

    Class that contains a Poincare Collection and references to matplotlib
    axes and is used to interactively pan through the collection's energy
    levels, as well as to display orbits by clicking on the surfaces of section

    Parameters
    ----------
    ax_sec: matplotlib.Axes
        Axes that should display the surfaces of section
    ax_orb: matplotlib.Axes
        Axes that should display the configuration space orbits
    data: PoincareCollection
        Collection that is imaged by the Tomography
    redraw_orbit: bool, optional
        If orbit k is selected at some energy, this parameter decides whether to
        redraw the new k-th orbit when the energy is changed. Not entirely physical
        but can be interesting to visualize. Disabled for performance.

    Attributes
    ----------
    [parameters]
    lines_sec: list of matplotlib.line2D
        List of length nb_orbits_per_E artists that draw the points of a surface
        of section. The points are updated every time the energy is changed.
    line_orb: matplotlib.line2D
        Artist that draws the orbit in ax_orb when a section is selected in ax_sec
    idx: int
        Index of the currently displayed energy level
    
    Figure layout:
    <0.03> margin <0.1> buttons <0.05> margin <0.36> axes <0.07> margin <0.36> axes <0.03> margin
    """
    def __init__(self,sections: np.ndarray ,orbitslist, zvclist: np.ndarray,energylist, mapper: PoincareMapper, figsize=(15,7), redraw_orbit: bool = True) -> None:
        """ Load Data """
        self._sl = sections
        self._ol = orbitslist
        self._zvcl = zvclist
        self._El = energylist
        self.mapper = mapper
        
        self._nEn = len(sections)
        self._nSec = len(sections[0])

        """ Construct figure """
        # Main fig & axes
        self.fig = plt.figure(figsize=figsize)
        aspct = figsize[0] / figsize[1]
        axw = 0.36
        self.ax_sec = self.fig.add_axes([0.18,0.08,axw,axw * aspct])
        self.ax_orb = self.fig.add_axes([0.61,0.08,axw,axw * aspct])
        self.ax_orb.axis('equal')
        ffs = 16
        self.ax_sec.set_xlabel("$x$",fontsize=ffs)
        self.ax_sec.set_ylabel("$\dot{x}$",fontsize=ffs)
        self.ax_orb.set_xlabel("$x$",fontsize=ffs)
        self.ax_orb.set_ylabel("$y$",fontsize=ffs)
        self.ax_sec.set_title("Section Plane",fontsize=ffs)
        self.ax_orb.set_title("Orbital Plane",fontsize=ffs)

        self.lines_sec = [self.ax_sec.plot([], [],'o',ms=0.3,color='black',picker=True,pickradius=5)[0] for i in range(self._nSec)]
        self.line_orb = self.ax_orb.plot([], [],lw=1,color='tomato')[0]
        self.line_zvc = self.ax_sec.plot([], [],lw=0.5,color='indianred')[0]

        # Quit button
        ax_quitbutton = self.fig.add_axes([0.03, 0.05, 0.1, 0.075])
        self.button_quit = Button(ax_quitbutton,"Quit",color='mistyrose',hovercolor='lightcoral')
        self.button_quit.on_clicked(self._quitfig)
        
        # Set period with a Text box
        ax_setperiod = self.fig.add_axes([0.08, 0.325, 0.05, 0.05])
        self.textbox_setperiod = TextBox(ax_setperiod,'$p=$  ',color='mistyrose',
                    hovercolor='lightcoral', initial=1)
        self.textbox_setperiod.on_submit(self._setperiod)
        self._p = 1

        # Button to enable/disable search mode
        ax_searchbutton = self.fig.add_axes([0.03, 0.4, 0.1, 0.075])
        self.button_search = Button(ax_searchbutton,'Search for\np-periodic orbits',
                    color='mistyrose',hovercolor='lightcoral')
        self.button_search.on_clicked(self._searchmode)
        self.line_psection, = self.ax_sec.plot([],[],'o',ms=4,color='mediumspringgreen')
        self._in_searchmode = False

        # Interactivity (switch energy, click on section)
        self._firstpick = True
        self.redraw_orbit = redraw_orbit
        self.idx = 0
        self.fig.canvas.mpl_connect('key_press_event',self)
        self._pickid = self.fig.canvas.mpl_connect('pick_event',self._onpick)
        self._in_redrawmode = False

        # Show lowest energy to start
        self.show(0)

        ### WIP
        ax_setredraw = self.fig.add_axes([0.08, 0.15, 0.05, 0.05])
        self.textbox_setredraw = TextBox(ax_setredraw,'$N=$  ',color='mistyrose',
                    hovercolor='lightcoral', initial=10)
        self._Nredraw = 10
        self.textbox_setredraw.on_submit(self._setredraw)

        ax_redrawbutton = self.fig.add_axes([0.03, 0.225, 0.1, 0.075])
        self.button_redraw = Button(ax_redrawbutton,'Redraw current view\n with N orbits',
                    color='mistyrose',hovercolor='lightcoral')
        self.button_redraw.on_clicked(self._redrawmode)
        """axredrawbtn = self.fig.add_axes([0.0, 0.15, 0.1, 0.075])
        self.redrawbtn = Button(axredrawbtn,'Redraw',color='mistyrose',hovercolor='lightcoral')
        self.redrawbtn.on_clicked(self._zoommode)
        self.fig.text(0.03,0.225,'Redraw current section\nwith N orbits:')
        axredrawtext = self.fig.add_axes([0.03, 0.15, 0.1, 0.05])
        self.redrawtext = TextBox(axredrawtext,'',
                                color='mistyrose',hovercolor='lightcoral',
                                initial=10)"""
        
    def __call__(self,event):
        """Interaction function to switch energy level by up/down keys
        Parameters
        ----------
        event : matplotlib.key_press_event
            Up or down key press event
        """
        
        ii = self.idx
        if event.key == 'up':
            if self._in_redrawmode:
                self._exitrdrw()
            ii += 1
        elif event.key == 'down':
            if self._in_redrawmode:
                self._exitrdrw()
            ii -= 1
        if ii in range(self._nEn):
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
        self.ax_sec.relim()
        self.ax_sec.autoscale()
        if not self._firstpick and self.redraw_orbit:
            self.line_orb.set_xdata(self._ol[idx][self.artistid][0])
            self.line_orb.set_ydata(self._ol[idx][self.artistid][1])
            self.ax_orb.relim()
            self.ax_orb.autoscale()
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
        self.line_orb.set_xdata(self._ol[self.idx][self.artistid][0])
        self.line_orb.set_ydata(self._ol[self.idx][self.artistid][1])
        self.ax_orb.relim()
        self.ax_orb.autoscale()
        self.fig.canvas.draw()
        self.prev_artist = event.artist

    def _quitfig(self,event):
        print("Program was exited by the user")
        plt.close(self.fig)
    def _redrawmode(self,event):
        if self._in_redrawmode:
            self._clearredraw()
        
        self._in_redrawmode = True
        for l in self.lines_sec:
            l.set_visible(False)
        # Set xlim to current zoom
        xl_phys = (np.amin(self._zvcl[self.idx,0]),np.amax(self._zvcl[self.idx,0]))
        xl_fig = self.ax_sec.get_xlim()
        vxl = self.ax_sec.get_ylim()
        xl = (max(xl_fig[0],xl_phys[0]),min(xl_fig[1],xl_phys[1]))
        print(xl)
        xdot0 = (vxl[0] + vxl[1])/2.

        print("Recalculating...")
        stmp,otmp,z = self.mapper.section(self._El[self.idx],N_orbits=self._Nredraw,N_points=len(self._sl[0][0][0]),nb_pts_orbit=None,xlim=xl,xdot0=xdot0)
        x = stmp[:,0,:].flatten()
        y = stmp[:,1,:].flatten()

        # Coloring attempt
        if 0:
            dx2 = np.diff(stmp[:,0,:],prepend=0)**2
            dy2 = np.diff(stmp[:,1,:],prepend=0)**2
            dists = np.sqrt(dx2+dy2).flatten()
            clr = dists
        else: clr = 'black'
        self.lstmp = self.ax_sec.scatter(x, y,s=0.3,c=clr)
        self.fig.canvas.draw()
    def _clearredraw(self):
        if hasattr(self,"lstmp"):
            self.lstmp.remove()
    def _exitrdrw(self):
        self._clearredraw()
        delattr(self,"lstmp")
        self._in_redrawmode = False
        for l in self.lines_sec:
            l.set_visible(True)
    def _setredraw(self,N):
        self._Nredraw = int(N)
    def _setperiod(self,p):
        self._p = int(p)
    def _searchmode(self,event):
        if not self._in_searchmode:
            self.button_search.color = 'firebrick'
            self._in_searchmode = True
            self._clickid = self.fig.canvas.mpl_connect('button_press_event',self._click)
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
    def _click(self,event):
        if event.inaxes == self.ax_sec:
            E = self._El[self.idx]
            q0 = [event.xdata,event.ydata]
            qstar = self.mapper.find_periodic_orbit(q0,E,
                    self._p,print_progress=True,eps=1e-3,maxiter=100)
            if qstar is not None:
                #self.line_psection.set_xdata(qstar[0])
                #self.line_psection.set_ydata(qstar[1])
                if 1:
                    eigvals = np.linalg.eigvals(self.mapper.jac(qstar,E,self._p))
                    print(eigvals)
                    print(np.abs(eigvals))
                s,o = self.mapper.integrate_orbit(qstar,E,N=5*self._p)
                self.line_psection.set_xdata(s[0])
                self.line_psection.set_ydata(s[1])
                self.line_orb.set_xdata(o[0])
                self.line_orb.set_ydata(o[1])
                self.line_orb.axes.relim()
                self.line_orb.axes.autoscale()
                self.fig.canvas.draw()
            else:
                print("The orbit finder did not converge with the provided starting guess")