# PoincarePy - Visualize Surfaces of Section Interactively
PoincarePy is a Python module to compute surfaces of section (or Poincaré maps) of 2D potentials and explore them interactively. The user can define a physical system by
using and combining a set of predefined potentials, or by implementing new ones, which is made very simple.

Installation
----------------------
### Requirements
- Numpy
- Matplotlib
- SciPy
- tqdm (Optional)

The install uses setuptools >=3.7. To install the package, simply clone the repo and run
`python3 -m pip install [path/to/repo]`
(preferably inside a virtual environment)

User Guide
----------------------
### Documentation & Tutorial
A general introduction as well as the full API documentation of the package can be found [here](https://poincarepy.readthedocs.io/en/latest/).
To help you get started, a tutorial series of 4 notebooks can be found in `tutorial_notebooks` and will guide you through the main features of the package.
A minimal guide is provided below.

### Minimal Guide
A typical workflow using `poincarepy` is:
1. Create a potential. This can be one of the predefined potentials or a new custom one, as well as combinations of the latter. For a choice of parameters, it is useful to viszalize the resulting potential, e.g. in a notebook, in order to get an intuition about the allowed energies and the scales of the dynamical variables. For example, when creating a logarithmic potential, this could look like
```python
import matplotlib.pyplot as plt
import poincarepy
logpot = poincarepy.potentials.LogarithmicPotential()
logpot.plot_x(-10,10)
plt.show()
```

2. Create a `PoincareMapper` object using the potential created above. This class deals with the computational part, and the other parameters are e.g. the maximum integration time (check the class source). With this object in hand, we can integrate orbits, generate Poincaré Maps, find periodic orbits, etc.

3. Generate a set of Poincaré sections in a range of energies. This is where step 1 is important, as its necessary to use an energy range that is compatible with the potential. Also, the `PoincareMapper.section` method will automatically find the limits of the zero-velocity-curve using a root-finding algorithm, but it still requires some bounds on those limits, even if very rough. Again, plotting the potential first is useful for this.

3. Create a `Tomography` object with the generated data. Upon creation, the object will open a figure that allows interactive exploration through the energy levels.
### Key Bindings
- `Up/Down`: Navigate through the energy levels
- `z`: Enable the single-orbit redrawing mode. Upon clicking on a point in phase space, a new orbit and its Poincaré Map are computed and displayed
- `t`: Enable the rectangle selection-redrawing mode. Same as above but a specified $N_{redraw}$ number of orbits inside a selected region are calculated.
