import pickle as pkl
from poincarepy import Tomography, PoincareCollection

col = PoincareCollection.load("example_collection.pkl")

tom = Tomography(col,title="Test")