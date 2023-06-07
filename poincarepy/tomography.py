import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RectangleSelector
import numpy as np
import pickle as pkl
from .collection import PoincareCollection
from .mapper import PoincareMapper

class Tomography:
    """Tomographic visualisation of an ensemble of surfaces of section at different energies

    Visualization class. Using precalculated data (Poincaré sections at different energies), allows
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
    collection : PoincareCollection
        The collection (i.e. set of surfaces of section at different energies) to visualize.
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
    def __init__(self,collection : PoincareCollection,
                 figsize=(12.5,6),title=None, redraw_orbit: bool = True,
                 axlabels=["$x$","$\dot{x}$","$x$","$y$"]) -> None:
        """ Load Data """

        self._sl = collection.sectionsarray
        self._ol = collection.orbitslist
        self._zvcl = collection.zvc_list
        self._El = collection.energylist
        self.mapper = collection.mapper

        self._nEn = len(self._sl)
        self._nSec = len(self._sl[0])

        # Precompute figure limits per energy
        x0, x1 = np.amin(self._zvcl[:,0],axis=1),np.amax(self._zvcl[:,0],axis=1)
        y0, y1 = np.amin(self._zvcl[:,1],axis=1),np.amax(self._zvcl[:,1],axis=1)
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
        self.ax_pot.set_title("$\phi(x,y=0)$")
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
            self.line_orb.set_xdata(self._ol[idx,self.artistid,0])
            self.line_orb.set_ydata(self._ol[idx,self.artistid,1])
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
        self._set_orb(self._ol[self.idx,self.artistid])
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
                    N_points_orbit=None,xlim=xl,xdot=xdot0,
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
        stmp,otmp,z = self.mapper.section(self._El[self.idx],xlim=(x1,x2),N_orbits=self._Nredraw,N_points=len(self._sl[0][0][0]),N_points_orbit=None,auto_lim=False,xdot=xdot0)
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