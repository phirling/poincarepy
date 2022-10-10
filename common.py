import numpy as np
import solver
from matplotlib.axes import Axes


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
    potential_info: string, optional
        Information about the potential that generated the collection

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
    
    def __init__(self,E_list,orbits_list,sections_list,potential_info="No information about potential") -> None:
        if not (len(E_list) == len(orbits_list) == len(sections_list)):
            raise ValueError("lists must be of the same length")
        self.energylist = E_list
        self.orbitslist = orbits_list
        self.sectionslist = sections_list
        self.potentialinfo = potential_info
        self.nb_energies = len(E_list)
        self.nb_orbits_per_E = len(orbits_list[0])
    def potential_info(self):
        """Print information about the potential that generated the collection"""
        print("Source Potential (sum if multiple):")
        print(self.potentialinfo)

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
    def __init__(self,ax_sec: Axes,ax_orb: Axes,data: PoincareCollection, redraw_orbit: bool = False) -> None:
        self._firstpick = True
        self.redraw_orbit = redraw_orbit
        self.collection = data
        self.lines_sec = [ax_sec.plot([], [],'o',ms=0.3,color='black',picker=True,pickradius=5)[0] for i in range(data.nb_orbits_per_E)]
        self.line_orb = ax_orb.plot([], [],lw=1)[0]
        self.idx = 0
        ax_orb.axis('equal')
        self._fig = ax_sec.figure
        self._fig.suptitle(data.potentialinfo)
        self._fig.canvas.mpl_connect('key_press_event',self)
        self._fig.canvas.mpl_connect('pick_event',self.onpick)
        self.show(0)
    def __call__(self,event):
        """Interaction function to switch energy level by up/down keys
        Parameters
        ----------
        event : matplotlib.key_press_event
            Up or down key press event
        """
        if event.key == 'up':
            ii = self.idx + 1
        elif event.key == 'down':
            ii = self.idx - 1
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
        self.lines_sec[0].axes.relim()
        self.lines_sec[0].axes.autoscale()
        if not self._firstpick and self.redraw_orbit:
            self.line_orb.set_xdata(self.collection.orbitslist[idx][self.artistid][0])
            self.line_orb.set_ydata(self.collection.orbitslist[idx][self.artistid][1])
            self.line_orb.axes.relim()
            self.line_orb.axes.autoscale()
        self.lines_sec[0].axes.figure.canvas.draw()
    def onpick(self,event):
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
        event.artist.set_color('red')
        event.artist.set_markersize(1.5)
        self.artistid = self.lines_sec.index(event.artist)
        self.line_orb.set_xdata(self.collection.orbitslist[self.idx][self.artistid][0])
        self.line_orb.set_ydata(self.collection.orbitslist[self.idx][self.artistid][1])
        self.line_orb.axes.relim()
        self.line_orb.axes.autoscale()
        event.artist.figure.canvas.draw()
        self.prev_artist = event.artist