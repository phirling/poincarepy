import pickle as pkl
from poincarepy import Tomography

with open("example_collection.pkl","rb") as f:
    col = pkl.load(f)

# TODO: have a constructor for PoincareCollection that takes a filename as argument
# TODO: same for Tomography but with a PoincareCollection as argument
mapper = col.mapper
pot = mapper.pot
orbits = col.orbitslist
sections = col.sectionsarray
energies = col.energylist
zvcs = col.zvc_list
tom = Tomography(sections,orbits,zvcs,energies,mapper,title="Test")