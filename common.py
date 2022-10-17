import numpy as np
import solver
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button, TextBox
from potentials import Potential
import copy
import scipy.optimize as scpopt

class PoincareMapper:
    def __init__(self,pot: Potential,crossing_function,
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
            raise ValueError("Starting point out of range")
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
    
    def __init__(self,E_list,orbits_list,sections_list,zvc_list,mapper: PoincareMapper) -> None:
        if not (len(E_list) == len(orbits_list) == len(sections_list)):
            raise ValueError("lists must be of the same length")
        self.energylist = E_list
        self.orbitslist = orbits_list
        self.sectionslist = sections_list
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
    
    """
    def __init__(self,ax_sec: Axes,ax_orb: Axes,data: PoincareCollection, redraw_orbit: bool = True) -> None:
        self._firstpick = True
        self.redraw_orbit = redraw_orbit
        self.collection = data
        self.lines_sec = [ax_sec.plot([], [],'o',ms=0.3,color='black',picker=True,pickradius=5)[0] for i in range(data.nb_orbits_per_E)]
        self.line_orb = ax_orb.plot([], [],lw=1,color='tomato')[0]
        self.line_zvc = ax_sec.plot([], [],lw=0.5,color='indianred')[0]
        self.idx = 0
        ax_orb.axis('equal')
        self._fig = ax_sec.figure
        self._fig.canvas.mpl_connect('key_press_event',self)
        self._pickid = self._fig.canvas.mpl_connect('pick_event',self._onpick)
        self.show(0)

        # Quit button
        ax_quitbutton = self._fig.add_axes([0.0, 0.05, 0.1, 0.075])
        self.button_quit = Button(ax_quitbutton,"Quit",color='mistyrose',hovercolor='lightcoral')
        self.button_quit.on_clicked(self._quitfig)
        
        """Interactive Periodic Orbit searching"""
        # Set period with a Text box
        ax_setperiod = self._fig.add_axes([0.05, 0.3, 0.05, 0.05])
        self.textbox_setperiod = TextBox(ax_setperiod,'$p$:  ',color='mistyrose',
                    hovercolor='lightcoral', initial=1)
        self.textbox_setperiod.on_submit(self._setperiod)

        # Button to enable/disable search mode
        ax_searchbutton = self._fig.add_axes([0.0, 0.375, 0.1, 0.075])
        self.button_search = Button(ax_searchbutton,'Search for\np-periodic orbits',
                    color='mistyrose',hovercolor='lightcoral')
        self.button_search.on_clicked(self._searchmode)
        
        # Initialize objects & attributes
        self.line_psection, = ax_sec.plot([],[],'o',ms=4,color='mediumspringgreen')
        self._p = 1
        self._in_searchmode = False

        ### WIP
        #axredrawbtn = self._fig.add_axes([0.0, 0.15, 0.1, 0.075])
        #self.redrawbtn = Button(axredrawbtn,'Redraw',color='mistyrose',hovercolor='lightcoral')
        self._fig.text(0.0,0.225,'Redraw current section\nwith N orbits:')
        axredrawtext = self._fig.add_axes([0.0, 0.15, 0.1, 0.05])
        self.redrawtext = TextBox(axredrawtext,'',
                                color='mistyrose',hovercolor='lightcoral',
                                initial=10)
        self.redrawtext.on_submit(self._redraw)
        
    def __call__(self,event):
        """Interaction function to switch energy level by up/down keys
        Parameters
        ----------
        event : matplotlib.key_press_event
            Up or down key press event
        """
        ii = self.idx
        if event.key == 'up':
            ii += 1
        elif event.key == 'down':
            ii -= 1
        if ii in range(self.collection.nb_energies):
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
            l.set_xdata(self.collection.sectionslist[idx][k][0])
            l.set_ydata(self.collection.sectionslist[idx][k][1])
        if hasattr(self.collection,'zvc_list'):
            self.line_zvc.set_xdata(self.collection.zvc_list[idx][0])
            self.line_zvc.set_ydata(self.collection.zvc_list[idx][1])
        self.lines_sec[0].axes.relim()
        self.lines_sec[0].axes.autoscale()
        if not self._firstpick and self.redraw_orbit:
            self.line_orb.set_xdata(self.collection.orbitslist[idx][self.artistid][0])
            self.line_orb.set_ydata(self.collection.orbitslist[idx][self.artistid][1])
            self.line_orb.axes.relim()
            self.line_orb.axes.autoscale()
        self.lines_sec[0].axes.set_title("E = {:.1f}".format(self.collection.energylist[idx]))
        self.lines_sec[0].axes.figure.canvas.draw()
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
        self.line_orb.set_xdata(self.collection.orbitslist[self.idx][self.artistid][0])
        self.line_orb.set_ydata(self.collection.orbitslist[self.idx][self.artistid][1])
        self.line_orb.axes.relim()
        self.line_orb.axes.autoscale()
        event.artist.figure.canvas.draw()
        self.prev_artist = event.artist

    def _quitfig(self,event):
        print("Program was exited by the user")
        plt.close(self._fig)
    def _redraw(self,N):
        new_N = int(N)
        pass
    def _setperiod(self,p):
        self._p = int(p)
    def _searchmode(self,event):
        if not self._in_searchmode:
            self.button_search.color = 'firebrick'
            self._in_searchmode = True
            self._clickid = self._fig.canvas.mpl_connect('button_press_event',self._click)
            self.line_orb.set_xdata([])
            self.line_orb.set_ydata([])
            self.line_psection.set_visible(True)
            self._fig.canvas.mpl_disconnect(self._pickid)
        else:
            self.button_search.color = 'mistyrose'
            self._in_searchmode = False
            self._fig.canvas.mpl_disconnect(self._clickid)
            self.line_psection.set_xdata([])
            self.line_psection.set_ydata([])
            self.line_orb.set_xdata([])
            self.line_orb.set_ydata([])
            self.line_psection.set_visible(False)
            self._pickid = self._fig.canvas.mpl_connect('pick_event',self._onpick)
    def _click(self,event):
        if event.inaxes == self.line_zvc.axes:
            E = self.collection.energylist[self.idx]
            q0 = [event.xdata,event.ydata]
            qstar = self.collection.mapper.find_periodic_orbit(q0,E,
                    self._p,print_progress=True,eps=1e-3)
            #print(qstar)
            if qstar is not None:
                #self.line_psection.set_xdata(qstar[0])
                #self.line_psection.set_ydata(qstar[1])
                if 1:
                    eigvals = np.linalg.eigvals(self.collection.mapper.jac(qstar,E,self._p))
                    print(eigvals)
                    print(np.abs(eigvals))
                s,o = self.collection.mapper.integrate_orbit(qstar,E,N=5*self._p)
                self.line_psection.set_xdata(s[0])
                self.line_psection.set_ydata(s[1])
                self.line_orb.set_xdata(o[0])
                self.line_orb.set_ydata(o[1])
                self.line_orb.axes.relim()
                self.line_orb.axes.autoscale()
                self._fig.canvas.draw()
            else:
                print("The orbit finder did not converge with the provided starting guess")