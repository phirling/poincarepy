import poincarepy as pcp

# To visualize a saved PoincareCollection, simply load the file and
# create a Tomography object from it!

col = pcp.PoincareCollection.load("example_collection.pkl")

tom = pcp.Tomography(col,title="Test")