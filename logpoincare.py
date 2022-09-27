from logging import root
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scpint
import argparse
from tqdm import tqdm
import solver

fs = (15,7)

parser = argparse.ArgumentParser(
    description="Poincare x vx Section of Logarithmic potential"
)
'''
The program has 2 modes: by default, no orbits are drawn and the program is in an interactive mode where
clicking on a point on the left plane (x,vx) sets the IC, draws an orbit and the corresponding Poincaré section.
If the --fill flag is passed, the program calculates a given number of orbits whose starting points lie uniformly
on the x axis (with vx=0) and then becomes interactive, where clicking on a poincaré curve shows its corresponding
orbit in xy-space on the right.

For the integration duration, the user can either (default) specify a number of crossings of the y plane (i.e. a number
of points in the Poincaré map), in which case the -tf argument is used as an upper limit or timeout to the integration time,
or the user can specify --no_count in which case the integration will keep going until tf is reached.
'''

parser.add_argument("-v0",type=float,default=10.,help="Characteristic Velocity")
parser.add_argument("-rc",type=float,default=1.,help="Characteristic Radius")
parser.add_argument("-q",type=float,default=0.8,help="Flattening Parameter")
parser.add_argument("-E",type=float,default=80,help="Energy of Orbit")

parser.add_argument("--jac",action='store_true',help="Compute the Jacobian")

parser.add_argument("-tf",type=float,default=100,help="Maximal integration time. If --no_count, this will be the obligatory final time")
parser.add_argument("--no_count",action='store_true',help="Integrate until t=tf, without counting the crossings")
parser.add_argument("-nb_crossings",type=int,default= 40,help="Terminate integration after n crossings of the plane (=nb of points in the Poincaré map)")
parser.add_argument("--fill",action='store_true',help="Try to fill the phase space with orbits")
parser.add_argument("-nb_orbs",type=int,default=11,help="Number of orbits to sample if --fill is passed")

args = parser.parse_args()

# Define Physical System
G = 4.299581e04
class LogarithmicPotential:
    def __init__(self,v0,rc,q):
        self.v0 = v0
        self.rc = rc
        self.q = q

    def potential(self,y):
        return 0.5*self.v0**2 * np.log(self.rc**2 + y[0]**2 + y[1]**2/self.q**2)

    def accel(self,y):
        A = - self.v0**2 / (self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
        ax = A*y[0]
        ay = A*y[1]/(self.q**2)
        return np.array([ax,ay])

    # Bounds on the zero velocity curve
    # xdot = ydot = y = 0
    def maxval_x(self,E):
        return np.sqrt(np.exp(2*E/self.v0**2)-self.rc**2)
    # xdot = ydot = x = 0
    def maxval_y(self,E):
        return np.sqrt(np.exp(2*E/self.v0**2)-self.rc**2) * self.q

pot = LogarithmicPotential(args.v0,args.rc,args.q)

# RHS of the system dy/dt = f(y) where y = (x,y,vx,vy)
def RHS(t,y):
    F = pot.accel(y)
    return [y[2],
            y[3],
            F[0],
            F[1]]

# Compute zero-velocity curve at E fixed
xlim = 0.9999*pot.maxval_x(args.E)
def xdot_of_x(x):
    Ed = args.E - pot.potential([x,0])
    if np.any(Ed < 0):
        raise ValueError("Outside of bounds")
    return np.sqrt(2*Ed)
xsp = np.linspace(-xlim,xlim,200)
xdotsp = xdot_of_x(xsp)

# Initialize Figure
# Left panel (ax[0]): x vx space with y=0 v_y > 0 : the Poincaré map space
# Right panel: configuration space (xy)
fig, ax = plt.subplots(1,2,figsize=fs)
ax[0].set_xlabel('$x$ [kpc]')
ax[0].set_ylabel('$v_x$ [km/s]')
ax[1].set_xlabel('$x$ [kpc]')
ax[1].set_ylabel('$y$ [kpc]')
ax[0].set_title("Poincaré section $y=0,v_y>0$")
ax[1].set_title("Orbit in $xy$-plane")
fig.suptitle('Logarithmic Potential, $q=$ {:.2f}, $E=$ {:.1f}'.format(args.q,args.E),fontsize=16)

# Zero velocity curve in Poincaré space
ax[0].plot(xsp,xdotsp)
ax[0].plot(xsp,-xdotsp)

# Text box to hold information
txt = ax[0].text(0.2, 0.9,'',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax[0].transAxes)

# Fix limit of the right panel
llim = 1.1*xlim
ax[1].set_xlim(-llim,llim)
ax[1].set_ylim(-llim,llim)

# Integration Parameters
tf = args.tf
t_span = (0,tf)
nb_points_orbit = 20000
t_eval = np.linspace(0,tf,nb_points_orbit) # <-- TODO: more efficient solution here
if args.no_count:
    event_count_max = None
else:
    event_count_max = args.nb_crossings

# Function that checks if y=0 with fixed direction of crossing (Vy>0)
def event_yplanecross(t,y):
    return y[1]
event_yplanecross.direction = 1

# Main program: either single-map interactive mode or auto-filled multi-map mode
# Mode 1:
if not args.fill:

    txt.set_text('Click on a point\nto start')

    # Create empty lines in both panels
    line_sections, = ax[0].plot([], [],'o',ms=1.5)
    line_orbits, = ax[1].plot([], [],lw=1)

    # Picking function used to get initial x, vx from user input
    def onpick(event):
        if event.inaxes == ax[0]:
            # x, xdot IC
            x, xdot = event.xdata, event.ydata
            # Orbit starts on y=0 plane
            y = 0

            # Calculate ydot from fixed E and x, xdot
            ED = 2*(args.E-pot.potential([x,y])) - xdot**2
            if ED >= 0:
                ydot = np.sqrt(ED)
                y0 = [x,y,xdot,ydot] 
                #res = scpint.solve_ivp(RHS,tsp,y0,t_eval=t_eval,method='DOP853',events=event_yplanecross)
                res = solver.integrate_orbit(RHS,t_span,y0,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)

                # Orbit Output
                xx = res['y'][0]
                yy = res['y'][1]

                # Ouput of event watcher
                yevs = res['y_events'][0]
                X = yevs[:,0]
                Xdot = yevs[:,2]

                # TODO: Compute jacobian etc
                if args.jac:
                    # One can view the poincaré section as a map from K --> K where K is the phase space. The map is the phase space
                    # position of the body at the moment it crosses the y=0 plane for the first time. Thus, poincare_map(y0) will
                    # always yield a result of the form [x,0,vx,vy]
                    def poincare_map(y0):
                        res = solver.integrate_orbit(RHS,t_span,y0,events=event_yplanecross,event_count_end=2)
                        return res['y_events'][0][1] # <- [0] selects the event type (only one), [1] is the first occurence (0 is the start)

                    # Most of the time, we are interested only in the restricted map T: (x,vx) --> (x,vx)
                    def poincare_map_restr(y0):
                        return poincare_map(y0)[[0,2]]

                    # The jacobian matrix of the map can be computed numerically by finite differences (back&forward)
                    def num_jacobian(y0,dx,dvx):
                        y0xf   = np.add(y0,[dx,0,0,0])      #[x+dx,y,xdot,ydot]
                        y0xb   = np.add(y0,[-dx,0,0,0])     #[x-dx,y,xdot,ydot]
                        y0vxf  = np.add(y0,[0,0,dvx,0])     #[x,y,xdot+dxdot,ydot]
                        y0vxb  = np.add(y0,[0,0,-dvx,0])    #[x,y,xdot-dxdot,ydot]

                        Txf = poincare_map_restr(y0xf)
                        Txb = poincare_map_restr(y0xb)
                        Tvxf = poincare_map_restr(y0vxf)
                        Tvxb = poincare_map_restr(y0vxb)

                        J00 = (Txf[0] - Txb[0]) / (2*dx)
                        J01 = (Tvxf[0] - Tvxb[0]) / (2*dxdot)
                        J10 = (Txf[1] - Txb[1]) / (2*dx)
                        J11 = (Tvxf[1] - Tvxb[1]) / (2*dxdot)

                        jac_matrix = np.array([[J00,J01],[J10,J11]])
                        return jac_matrix

                    ddiv = 10000.
                    dx = abs(x/ddiv)
                    dxdot = abs(xdot/ddiv)

                    '''
                    Version with only forward
                    y01 = [x+dx,y,xdot,ydot]
                    y02 = [x,y,xdot+dxdot,ydot]
                    res01 = scpint.solve_ivp(RHS,t_span,y01,method='DOP853',events=event_yplanecross)
                    res02 = scpint.solve_ivp(RHS,t_span,y02,method='DOP853',events=event_yplanecross)

                    yevs01 = res01['y_events'][0]
                    yevs02 = res02['y_events'][0]

                    ii = 1
                    J00 = (yevs01[:,0][ii] - yevs[:,0][ii])/ dx
                    J01 = (yevs01[:,2][ii] - yevs[:,2][ii])/ dx
                    J10 = (yevs02[:,0][ii] - yevs[:,0][ii])/ dxdot
                    J11 = (yevs02[:,2][ii] - yevs[:,2][ii])/ dxdot

                    Jac = np.array([[J00,J01],[J10,J11]])
                    print(Jac)
                    

                    y0xf   = [x+dx,y,xdot,ydot]
                    y0xb   = [x-dx,y,xdot,ydot]
                    y0vxf  = [x,y,xdot+dxdot,ydot]
                    y0vxb  = [x,y,xdot-dxdot,ydot]
                    resxf  = solver.integrate_orbit(RHS,t_span,y0xf,events=event_yplanecross,event_count_end=2)
                    resxb  = solver.integrate_orbit(RHS,t_span,y0xb,events=event_yplanecross,event_count_end=2)
                    resvxf = solver.integrate_orbit(RHS,t_span,y0vxf,events=event_yplanecross,event_count_end=2)
                    resvxb = solver.integrate_orbit(RHS,t_span,y0vxb,events=event_yplanecross,event_count_end=2)

                    yevsxf  = resxf['y_events'][0]
                    yevsxb  = resxb['y_events'][0]
                    yevsvxf = resvxf['y_events'][0]
                    yevsvxb = resvxb['y_events'][0]

                    ii = 1 # <- after 1 crossing
                    J00 = (yevsxf[ii,0] - yevsxb[ii,0])/(2*dx)
                    J01 = (yevsvxf[ii,0] - yevsvxb[ii,0])/(2*dxdot)
                    J10 = (yevsxf[ii,2] - yevsxb[ii,2])/(2*dx)
                    J11 = (yevsvxf[ii,2] - yevsvxb[ii,2])/(2*dxdot)

                    #J00 = (yevs01[:,0][ii] - yevs[:,0][ii])/ dx
                    #J01 = (yevs01[:,2][ii] - yevs[:,2][ii])/ dx
                    #J10 = (yevs02[:,0][ii] - yevs[:,0][ii])/ dxdot
                    #J11 = (yevs02[:,2][ii] - yevs[:,2][ii])/ dxdot

                    #print(yevsxf[ii,0])
                    Jac = np.array([[J00,J01],[J10,J11]])
                    #print(Jac)
                    '''
                    #J = num_jacobian(y0,dx,dxdot)
                    #jacnum_det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
                    #print(jacnum_det)

                # Plot Poincaré section & Orbit in xy space
                line_sections.set_xdata(X)
                line_sections.set_ydata(Xdot)

                line_orbits.set_xdata(xx)
                line_orbits.set_ydata(yy)

                # Show info
                txt.set_text('IC:\n$x=$ {:.2f} kpc\n$v_x=$ {:.2f} km/s\n$v_y=$ {:.2f} km/s'.format(x,xdot,ydot))
                fig.canvas.draw()
            else:
                print("Point outside of zero-velocity curve")

    fig.canvas.mpl_connect('button_press_event', onpick)

# Mode 2: Draw a set of orbits in advance, show xy-orbit by clicking
else:
    # Create empty line in the right panel (the left panel will be filled)
    line_orbits, = ax[1].plot([], [],lw=1)
    
    # Interactivity is handled by a class (cleaner & allows some visual touches like colorizing the picked curve)
    class OrbitPicker:
        def __init__(self,line_orbits):
            self.line_orbits = line_orbits
            self.sections = [] # <- list of line2d objects: the poincaré sections in the left panel
            self.orbits_x = [] # <- list of arrays containing the x/y coords of the orbits
            self.orbits_y = [] #    These are plotted when the corresponding section in the left panel is clicked
            self.firstpick = True
            line_orbits.figure.canvas.mpl_connect('pick_event',self)

        def append_orbit(self,x,y):
            self.orbits_x.append(x)
            self.orbits_y.append(y)
        
        def append_section(self,secline):
            self.sections.append(secline)
        
        def __call__(self,event):
            if self.firstpick:
                txt.set_text('')
                self.firstpick = False
            else:
                self.prev_artist.set_color('black')
                self.prev_artist.set_markersize(1)

            event.artist.set_color('turquoise')
            event.artist.set_markersize(2)
            k = event.artist.id # <- When the section is plotted (see below), an "ID" attribute is set to
                                #    match it with its corresponding orbit. TODO: find a clearer approach?
            self.line_orbits.set_xdata(self.orbits_x[k])
            self.line_orbits.set_ydata(self.orbits_y[k])
            self.line_orbits.figure.canvas.draw()
            self.prev_artist = event.artist

    # Sample ICs uniformly on x axis
    x_ic = np.linspace(-xlim,xlim,args.nb_orbs)
    ydot_ic = np.empty(args.nb_orbs)
    for i,x in enumerate(x_ic):
        ydot_ic[i] = np.sqrt(2*(args.E-pot.potential([x_ic[i],0])))
    
    
    txt.set_text('Click on a curve to\nshow its orbit')
    Picker = OrbitPicker(line_orbits)

    # Integrate the nb_orbs trajectories
    for k in tqdm(range(args.nb_orbs)):
        y0 = [x_ic[k],0,0,ydot_ic[k]]
        #res = scpint.solve_ivp(RHS,tsp,y0,t_eval=t_eval,method='DOP853',events=event_yplanecross)
        res = solver.integrate_orbit(RHS,t_span,y0,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)

        # Output the orbits to be displayed upon clicking
        Picker.append_orbit(res['y'][0],res['y'][1])

        # Ouput of event watcher
        yevs = res['y_events'][0]
        X = yevs[:,0]
        Xdot = yevs[:,2]

        # Plot Poincaré curve and set its ID to pair it with corresponding orbit
        secline, = ax[0].plot(X, Xdot,'o',ms=0.7,color='black',picker=True)
        secline.id = k
        Picker.append_section(secline)

plt.show()