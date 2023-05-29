import numpy as np
import pickle as pkl
from .mapper import PoincareMapper

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
    
    def save(self,fname = "pcollection.pkl"):
        with open(fname,"wb") as f:
            pkl.dump(self,f)
    
    @classmethod
    def load(cls,fname):
        with open(fname,"rb") as f:
            old = pkl.load(f)
        return cls(old.energylist,old.orbitslist,old.sectionsarray,old.zvc_list,old.mapper)