import numpy as np
import argparse
import pickle as pkl
from poincarepy import PoincareCollection, Tomography, PoincareMapper
import poincarepy.potentials as potentials

# Example script to compute & visualize Poincaré sections in a rotating logarithmic potential
# like the one presented during lecture 8 of Astro III at EPFL
# 
# You can run the script with default parameters (see below) or specify values for
# the potential's parameters, the properties of the Poincaré sections & so on
#
# === Troubleshooting ===
# "No roots found, use larger Nsteps": plot the potential with the parameters you use and look at
# the typical x-scale for the energies you consider. Set xlim (below) to this scale (very broadly)
# Explanation: in order to automatically find the xlims of the zero-velocity curves, the algorithm
# needs a rough estimate on where to look for them
#
# "Tmax was reached before N crossings cound occur": Increase the parameter -tmax.
#  Explanation: the integrator uses an upper limit in integration time to reach the desired
#  number of map crossings
#
# The configuration space orbits look pointy and not really smooth: Decrease the parameter -tmax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poincare x,vx Section of the rotating logarithmic potential")

    # Potential Parameters
    parser.add_argument("-v0",type=float,default=10.,help="Characteristic velocity of the log potential")
    parser.add_argument("-rc",type=float,default=0.14,help="Characteristic radius of the log potential")
    parser.add_argument("-q",type=float,default=0.8,help="Flattening of the log potential")
    parser.add_argument("-omega",type=float,default=0.,help="Rotation frequency")

    # Poincaré Section Parameters
    parser.add_argument("-N_points",type=int,default= 40,help="Number of points in the sections")
    parser.add_argument("-N_orbits",type=int,default=15,help="Number of orbits in each section")
    parser.add_argument("-Emin",type=float,default=30., help="Lowest energy to compute")
    parser.add_argument("-Emax",type=float,default=240., help="Highest energy to compute")
    parser.add_argument("-N_E",type=int,default=20,help="Number of energy levels")

    # Integration Parameters (increase this value if you get a RuntimeError that tmax was reached)
    parser.add_argument("-tmax",type=float,default=200,help="Maximal integration time. If --no_count, this will be the obligatory final time")
    
    # Import/Export Parameters
    parser.add_argument("-save",type=str,default=None)
    parser.add_argument("-open",type=str,default=None)

    # Miscellanous Parameters
    parser.add_argument("--no_orbit_redraw",action='store_false')
    parser.add_argument("--progress",action='store_true',help="Use tqdm to show progress bar")

    args = parser.parse_args()
    if not (0. < args.q < 1.):
        raise ValueError("q must be in the interval [0,1]")


    # Compute Poincaré Map (run without loading previous pkl file)
    if args.open is None:

        # Define the potential
        logpot = potentials.LogarithmicPotential(v0=args.v0,rc=args.rc,q=args.q)
        rotpot = potentials.zRotation(omega=args.omega)
        pot = potentials.CombinedPotential(logpot,rotpot)

        # Mapper object to do computations (see https://poincarepy.readthedocs.io/en/latest/api.html)
        mapper = PoincareMapper(pot,max_integ_time=args.tmax)

        # Print some info
        print("Generating {:n} Poincare maps in the potential:\n".format(args.N_E))
        print(pot.info())
        print("from E={:.2f} to E={:.2f}, with {:n} orbits per map and {:n} points (crossings) per orbit.".format(args.Emin,args.Emax,args.N_orbits,args.N_points))
        print("Maximum integration time is t_max={:.2f}".format(args.tmax),end="")
        if args.save is not None:
            print(", output will be saved to " + args.save + ".")
        else:
            print(", output will not be saved.")
        
        # Create Poincare sections over the specified range of energies
        energies = np.linspace(args.Emin,args.Emax,args.N_E)
        xlim = [-20,20] # <- This is a very broad range in which the code
                        # automatically searches for the true zero-velocity curve limits 
        sections, orbits, zvcs = mapper.section_collection(energies,xlim,args.N_orbits,args.N_points)
        
        # Create PoincareCollection object for convenient pickling
        col = PoincareCollection(energies,orbits,sections,zvcs,mapper)

        if args.save is not None:
            with open(args.save,'wb') as f:
                pkl.dump(col,f)

    # If a computation was done and saved previously, you can import it with the -open flag
    else:
        with open(args.open,'rb') as f:
            col = pkl.load(f)
    
        mapper = col.mapper
        pot = mapper.pot
        orbits = col.orbitslist
        sections = col.sectionsarray
        energies = col.energylist
        zvcs = col.zvc_list
    
    # The computed data (Poincaré sections & corresponding orbits) are visualized with a Tomography object (see API)
    ttl = "Rotating Logarithmic Potential\n$r_c={:.1f}, v_0={:.1f}, q={:.2f}, \omega={:.1f}$".format(args.rc,args.v0,args.q,args.omega)
    tom = Tomography(sections,orbits,zvcs,energies,mapper,title=ttl)