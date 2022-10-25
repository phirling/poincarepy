import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
from common import PoincareCollection, Tomography, PoincareMapper
from potentials import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poincare x vx Section")
    # Integration Parameters (Collection parameters)
    parser.add_argument("-tmax",type=float,default=200,help="Maximal integration time. If --no_count, this will be the obligatory final time")
    parser.add_argument("-N_points",type=int,default= 40,help="Terminate integration after n crossings of the plane (=nb of points in the Poincaré map)")
    parser.add_argument("-N_orbits",type=int,default=15,help="Number of orbits to sample if --fill is passed")

    # Tomography Parameters
    parser.add_argument("-Emin",type=float,default=30.)
    parser.add_argument("-Emax",type=float,default=240.)
    parser.add_argument("-N_E",type=int,default=20,help="Number of energy slices in tomographic mode")
    parser.add_argument("--no_orbit_redraw",action='store_false')

    # Script Parameters
    parser.add_argument("--progress",action='store_true',help="Use tqdm to show progress bar")
    parser.add_argument("-save",type=str,default=None)
    parser.add_argument("-open",type=str,default=None)

    args = parser.parse_args()


    # Run without loading previous pkl file
    if args.open is None:

        # Define a potential
        r0 = (0,0)
        logpot = LogarithmicPotential(zeropos=r0)
        rotpot = zRotation(0.3,zeropos=r0)
        plumpot = PlummerPotential(zeropos=r0)
        pot = CombinedPotential(logpot,rotpot)
        #pot = CombinedPotential(plumpot,rotpot)
        #pot = CombinedPotential(plumpot,logpot,rotpot)
        # ...

        # Mapper with default parameters for integration time etc
        mapper = PoincareMapper(pot,max_integ_time=args.tmax)

        # Print some info
        print("Generating {:n} Poincare maps in the potential:\n".format(args.N_E))
        print(pot.info())
        print("from E={:.2f} to E={:.2f}, with {:n} orbits per map and {:n} points (crossings) per map.".format(args.Emin,args.Emax,args.N_orbits,args.N_points))
        print("Maximum integration time is t_max={:.2f}".format(args.tmax),end="")
        if args.save is not None:
            print(", output will be saved to " + args.save + ".")
        else:
            print(", output will not be saved.")
        
        # Create Poincare sections over a range of energies
        energylist = np.linspace(args.Emin,args.Emax,args.N_E)
        orbitslist = []
        sectionslist = np.empty((args.N_E,args.N_orbits,2,args.N_points))
        zvclist = np.empty((args.N_E,2,800))
        for j,e in enumerate(energylist):
            print("Map #{:n} at E = {:.2f}".format(j,e))
            s,o,zvc = mapper.section(e,args.N_orbits,args.N_points)
            orbitslist.append(o)
            sectionslist[j] = s
            zvclist[j] = zvc
        
        # Create PoincareCollection object for convenient pickling
        col = PoincareCollection(energylist,orbitslist,sectionslist,zvclist,mapper)

        if args.save is not None:
            with open(args.save,'wb') as f:
                pkl.dump(col,f)

    # Run by loading previous pkl file
    else:
        with open(args.open,'rb') as f:
            col = pkl.load(f)
    
        mapper = col.mapper
        orbitslist = col.orbitslist
        sectionslist = col.sectionsarray
        energylist = col.energylist
        zvclist = col.zvc_list
    
    """ Show Results"""
    tom = Tomography(sectionslist,orbitslist,zvclist,energylist,mapper)