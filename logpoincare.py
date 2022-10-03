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

To set the integration time, the user can either (default) specify a number of crossings of the y plane (i.e. a number
of points in the Poincaré map), in which case the -tf argument is used as an upper limit or timeout to the integration time,
or the user can specify --no_count in which case the integration will keep going until tf is reached.

== Work in progress: ==
The --jac flag in interactive mode: for calculating the jacobian and eventually find periodic orbits

'''

parser.add_argument("-mode",type=str,default='interactive',help="Program mode (interactive, orbitfinder, fill, tomographic)")
parser.add_argument("-v0",type=float,default=10.,help="Characteristic Velocity")
parser.add_argument("-rc",type=float,default=1.,help="Characteristic Radius")
parser.add_argument("-q",type=float,default=0.8,help="Flattening Parameter")
parser.add_argument("-E",type=float,default=80,help="Energy of Orbit")

parser.add_argument("--jac",action='store_true',help="Compute the Jacobian & find periodic orbits near starting point")

parser.add_argument("-tf",type=float,default=100,help="Maximal integration time. If --no_count, this will be the obligatory final time")
parser.add_argument("--no_count",action='store_true',help="Integrate until t=tf, without counting the crossings")
parser.add_argument("-nb_crossings",type=int,default= 40,help="Terminate integration after n crossings of the plane (=nb of points in the Poincaré map)")
parser.add_argument("--fill",action='store_true',help="Try to fill the phase space with orbits")
parser.add_argument("-nb_orbs",type=int,default=11,help="Number of orbits to sample if --fill is passed")

# Tomographic mode
parser.add_argument("-Emin",type=float,default=20)
parser.add_argument("-Emax",type=float,default=200)
parser.add_argument("-nb_E",type=int,default=5,help="Number of energy slices in tomographic mode")
parser.add_argument("--save",action='store_true')
parser.add_argument("-open",type=str,default=None)

args = parser.parse_args()

# Define Physical System. The potential is implemented as a class in view of modularity,
# since eventually multiple potentials may be added
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
#ax[0].plot(xsp,xdotsp)
#[0].plot(xsp,-xdotsp)

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
t_eval = np.linspace(0,tf,nb_points_orbit) # ts at which the output of the integrator is stored.
                                           # in reality, integration normally stops before tf
                                           # (if a the given number of crossings is reached)
                                           # and so a large part of these ts will not be used.
# Decide whether to terminate integration after N crossings or integrate until tf
if args.no_count:
    event_count_max = None
else:
    event_count_max = args.nb_crossings

# Function that checks if y=0 with fixed direction of crossing (Vy>0)
def event_yplanecross(t,y):
    return y[1]
event_yplanecross.direction = 1

# To calculate vy from x,vx
def vy(x,vx):
    ED = 2*(args.E-pot.potential([x,0])) - vx**2
    if ED < 0:
        return None
    else:
        return np.sqrt(ED)




''' ###### Main program: either single-map interactive mode (1) or auto-filled multi-map mode (2) ####### '''

# Mode 1:
if args.mode == "interactive" or args.mode == "orbitfinder":
    txt.set_text('Click on a point\nto start')

    # Create empty lines in both panels
    line_sections, = ax[0].plot([], [],'o',ms=1.5)
    line_orbits, = ax[1].plot([], [],lw=1)

    # Picking function used to get initial x, vx from user input
    def onpick(event):
        if event.inaxes == ax[0]:
            x, xdot = event.xdata, event.ydata
            ydot = vy(x,xdot)

            if ydot is not None:
                y0 = [x,0,xdot,ydot] 
                #res = scpint.solve_ivp(RHS,tsp,y0,t_eval=t_eval,method='DOP853',events=event_yplanecross) # old integrator (scipy)
                res = solver.integrate_orbit(RHS,t_span,y0,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)

                # Orbit Output
                xx = res['y'][0]
                yy = res['y'][1]

                # Ouput of event watcher
                yevs = res['y_events'][0]
                X = yevs[:,0]
                Xdot = yevs[:,2]

                # Plot Poincaré section & Orbit in xy space
                line_sections.set_xdata(X)
                line_sections.set_ydata(Xdot)

                line_orbits.set_xdata(xx)
                line_orbits.set_ydata(yy)

                # Show info
                txt.set_text('IC:\n$x=$ {:.2f} kpc\n$v_x=$ {:.2f} km/s\n$v_y=$ {:.2f} km/s'.format(x,xdot,ydot))
                fig.canvas.draw()

                # TODO: THIS SECTION IS FOR TESTING THE JACOBIAN COMPUTATION
                if args.mode == "orbitfinder":
                    # One can view the poincaré section as a map from K --> K where K is the phase space. The map is the phase space
                    # position of the body at the moment it crosses the y=0 plane after one turn. Since y=0 in K and vy is determinable,
                    # in reality the poincare map can be seen as a 2D map from (x,vx) to (x,vx)
                    def poincare_map(q): # here q = [x,vx] is a 2d vector
                        ydot = vy(q[0],q[1])
                        if ydot is not None:
                            y0 = [q[0],0,q[1],ydot] # <- the integrator still needs a 4d vector
                            res = solver.integrate_orbit(RHS,t_span,y0,events=event_yplanecross,event_count_end=2)
                            # [0] selects the event type (only one), [1] is the first occurence (0 is the start), [[0,2]] returns only x,vx
                            return res['y_events'][0][1][[0,2]] 
                        else:
                            raise ValueError("blabla")
                    
                    # Jacobian of the 2d poincare map using finite differences
                    def jacobian_num(x,vx,dx,dvx):
                        Txf = poincare_map([x+dx,vx])
                        Txb = poincare_map([x-dx,vx])
                        Tvxf = poincare_map([x,vx+dvx])
                        Tvxb = poincare_map([x,vx-dvx])

                        J00 = (Txf[0] - Txb[0]) / (2*dx)
                        J01 = (Tvxf[0] - Tvxb[0]) / (2*dxdot)
                        J10 = (Txf[1] - Txb[1]) / (2*dx)
                        J11 = (Tvxf[1] - Tvxb[1]) / (2*dxdot)

                        jac_matrix = np.array([[J00,J01],[J10,J11]])
                        return jac_matrix
                    
                    # We also implement the poincare_map method in the full 4d space for future convenience
                    def poincare_map_4d(y):
                        res = solver.integrate_orbit(RHS,t_span,y,events=event_yplanecross,event_count_end=2)
                        return res['y_events'][0][1] # <- [0] selects the event type (only one), [1] is the first occurence (0 is the start)

                    # Example values for finite differences
                    ddiv = 10000.
                    dx = abs(x/ddiv)
                    dxdot = abs(xdot/ddiv)

                    ###### TESTS ######
                    q = np.array([x,xdot])
                    F = lambda q: poincare_map(q) - q
                    dF = lambda q: jacobian_num(q[0],q[1],dx,dxdot) - np.identity(2)
                    
                    import scipy.optimize as opt

                    # Test1: The nnls method does not converge
                    #def delta(q):
                    #    F = poincare_map(q) - q
                    #    dF = jacobian_num(q[0],q[1],dx,dxdot) - np.identity(2)
                    #    delta, resid = opt.nnls(dF,-F)
                    #    return delta,resid

                    # Test2: The scipy root finding methods work, but only find the solution at the origin
                    #print(opt.root(F,q,jac=dF))
                    #print(opt.fsolve(F,q,fprime=dF))

                    # Working method: use lsq_linear to find dq as in Pfenniger92
                    def delta_qn(q):
                        A = dF(q)
                        b = F(q)
                        return opt.lsq_linear(A,-b)

                    EPS = 1e-5

                    def find_periodic_orbit(q0,eps,maxiter):
                        ii = 0
                        qn = q0
                        deltq = q
                        while np.linalg.norm(deltq) > eps:
                            ii += 1
                            deltq = delta_qn(qn)['x']
                            qn += deltq
                            if ii > maxiter:
                                print("Maximum number of iterations reached")
                                break
                        print("Converged to a periodic orbit after " + str(ii) + " iterations:")
                        print("[x,vx] = [{:.3e},{:.3e}]".format(qn[0],qn[1]))
                        txt.set_text('Converged to a periodic orbit\nafter {:.0f} iterations:\n[x,vx] = [{:.1e},{:.1e}]'.format(ii,qn[0],qn[1]))
                        ax[1].set_title('Found periodic orbit')
                        #print("[x,y,vx,vy] = [{:.3e},0,{:.3e},{:.3e}]".format(qn[0],qn[1],vy(qn[0],qn[1])))
                        return qn
                    
                    q_per = find_periodic_orbit(q,EPS,100)

                    # Plot result
                    yq = [q_per[0],0.,q_per[1],vy(q_per[0],q_per[1])]
                    resnew = solver.integrate_orbit(RHS,t_span,yq,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)
                    xx = resnew['y'][0]
                    yy = resnew['y'][1]
                    yevs = resnew['y_events'][0]
                    X = yevs[:,0]
                    Xdot = yevs[:,2]
                    line_sections.set_xdata(X)
                    line_sections.set_ydata(Xdot)
                    line_orbits.set_xdata(xx)
                    line_orbits.set_ydata(yy)

                    
                    fig.canvas.draw()
            else:
                print("Point outside of zero-velocity curve")

    fig.canvas.mpl_connect('button_press_event', onpick)

# Mode 2: Draw a set of orbits in advance, show xy-orbit by clicking
elif args.mode == "fill": # Actually just a special case of tomographic
    class OrbitPicker:
        def __init__(self,ax_poincare,ax_orbit,E):
            self.line_orbits, = ax_orbit.plot([], [],lw=1)
            self.sections = [] # <- list of line2d objects: the poincaré sections in the left panel
            self.orbits_x = [] # <- list of arrays containing the x/y coords of the orbits
            self.orbits_y = [] #    These are plotted when the corresponding section in the left panel is clicked
            self.firstpick = True
            self.E = E
            ax_poincare.figure.canvas.mpl_connect('pick_event',self)

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
                self.prev_artist.set_markersize(0.3)

            event.artist.set_color('red')
            event.artist.set_markersize(1.5)
            k = event.artist.id # <- When the section is plotted (see below), an "ID" attribute is set to
                                #    match it with its corresponding orbit. TODO: find a clearer approach?
            self.line_orbits.set_xdata(self.orbits_x[k])
            self.line_orbits.set_ydata(self.orbits_y[k])
            self.line_orbits.figure.canvas.draw()
            self.prev_artist = event.artist
            
        def integrate(self):
            # Sample ICs uniformly on x axis
            x_ic = np.linspace(-xlim,xlim,args.nb_orbs)
            ydot_ic = np.sqrt(2*(self.E-pot.potential([x_ic,np.zeros(args.nb_orbs)])))
            
            for k in range(args.nb_orbs):
                y0 = [x_ic[k],0,0,ydot_ic[k]]
                res = solver.integrate_orbit(RHS,t_span,y0,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)

                # Output the orbits to be displayed upon clicking
                self.append_orbit(res['y'][0],res['y'][1])

                # Ouput of event watcher
                yevs = res['y_events'][0]
                X = yevs[:,0]
                Xdot = yevs[:,2]

                # Plot Poincaré curve and set its ID to pair it with corresponding orbit
                secline, = ax[0].plot(X, Xdot,'o',ms=0.3,color='black',picker=True)
                secline.id = k
                self.append_section(secline)

    Picker = OrbitPicker(ax[0],ax[1],args.E)
    Picker.integrate()

elif args.mode == "tomographic":
    class OrbitEnsemble:
        def __init__(self,potential,lines_poincare,line_orbit,E,integrate=True):
            self.E = E
            self.N_orbits = len(lines_poincare)
            self.line1 = lines_poincare
            self.line2 = line_orbit
            self.orbits = []
            self.sections = []
            self.pot = potential
            if integrate: self.integrate()
            self.firstpick = True
        def integrate(self):
            xlim = 0.9999*self.pot.maxval_x(self.E)
            x_ic = np.linspace(-xlim,xlim,self.N_orbits)
            ydot_ic = np.sqrt(2*(self.E-self.pot.potential([x_ic,np.zeros(self.N_orbits)])))
            for k in range(self.N_orbits):
                y0 = [x_ic[k],0,0,ydot_ic[k]]
                res = solver.integrate_orbit(RHS,t_span,y0,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)

                self.orbits.append(res['y'])
                yevs = res['y_events'][0].T
                self.sections.append(yevs[[0,2]])
        def show(self):
            if self.sections: # Don't show anything if empty
                for i in range(self.N_orbits):
                    self.line1[i].set_xdata(self.sections[i][0])
                    self.line1[i].set_ydata(self.sections[i][1])
                self.line1[0].axes.relim()
                self.line1[0].axes.autoscale()
                self.line1[0].figure.canvas.draw()
                self.cid = self.line1[0].figure.canvas.mpl_connect('pick_event',self)
        def hide(self):
            if not self.firstpick:
                self.prev_artist.set_color('black')
                self.prev_artist.set_markersize(0.3)
            self.line1[0].figure.canvas.mpl_disconnect(self.cid)
        def __call__(self,event):
            if self.firstpick:
                self.firstpick = False
            else:
                self.prev_artist.set_color('black')
                self.prev_artist.set_markersize(0.3)
            event.artist.set_color('red')
            event.artist.set_markersize(1.5)
            
            artistidx = self.line1.index(event.artist)
            self.line2.set_xdata(self.orbits[artistidx][0])
            self.line2.set_ydata(self.orbits[artistidx][1])
            self.line2.axes.axis('equal')
            self.line2.axes.relim()
            self.line2.axes.autoscale()
            event.artist.figure.canvas.draw()
            self.prev_artist = event.artist
        def set_data(self,sections,orbits):
            # check if not empty
            self.orbits = orbits
            self.sections = sections
        def get_data(self):
            return self.sections, self.orbits

    class TomographicPlot:
        def __init__(self,fig,ensemblelist):
            self.idx = 0
            self.ensemblelist = ensemblelist
            self.ensemblelist[self.idx].show()
            fig.canvas.mpl_connect('key_press_event',self)
        def __call__(self,event):
            if event.key == 'up':
                ii = self.idx + 1
            elif event.key == 'down':
                ii = self.idx - 1
            if ii in range(len(self.ensemblelist)):
                self.ensemblelist[self.idx].hide()
                self.ensemblelist[ii].show()
                self.idx = ii
        def get_data(self):
            data = []
            for oe in self.ensemblelist:
                data.append(oe.get_data())
            return data
        def set_data(self,data):
            for i,oe in enumerate(ensemblelist):
                oe.set_data(data[i][0],data[i][1])

    # Format to output/input data:
    # list with dim 0 == ensemble index
    # dim 1: 0 == sections, 1 == orbits
    # dim 2 == number of orbit in ensemble
    ##### TESTS
    import pickle as pkl
    # Note: pickling does not work because of figure/axes links. Do not use this function yet
    # Create empty line objects in the figure to speed up switching between E frames
    lines_sections = [ax[0].plot([], [],'o',ms=0.3,color='black',picker=True)[0] for i in range(args.nb_orbs)]
    line_orbit = ax[1].plot([], [],lw=1)[0]
    E_range = np.linspace(args.Emin,args.Emax,args.nb_E)
    ensemblelist = []
    is_new = args.open is None
    print("Generating plots for {:.0f} energy levels...".format(args.nb_E))
    for e in tqdm(E_range):
        ensemblelist.append(OrbitEnsemble(pot,lines_sections,line_orbit,e,is_new))

    tom = TomographicPlot(fig,ensemblelist)
    if not is_new:
        with open(args.open,'rb') as f:
            imported_data = pkl.load(f)
        tom.set_data(imported_data)
        tom.ensemblelist[0].show()
    
    if args.save:
        with open("ensemblelist.pkl",'wb') as f:
            pkl.dump(tom.get_data(),f)

elif args.mode == "test":
    pass
else:
    raise NameError("Unknown mode: '{:s}'.".format(args.mode))
plt.show()



'''
---------- Old code ---------
# One can view the poincaré section as a map from K --> K where K is the phase space. The map is the phase space
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
'''
