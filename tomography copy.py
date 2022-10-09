import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import solver

fs = (15,7)

parser = argparse.ArgumentParser(
    description="Poincare x vx Section of Logarithmic potential"
)

parser.add_argument("-tf",type=float,default=100,help="Maximal integration time. If --no_count, this will be the obligatory final time")
parser.add_argument("-nb_crossings",type=int,default= 40,help="Terminate integration after n crossings of the plane (=nb of points in the Poincaré map)")
parser.add_argument("-nb_orbs",type=int,default=11,help="Number of orbits to sample if --fill is passed")

parser.add_argument("-v0",type=float,default=10.,help="Characteristic Velocity")
parser.add_argument("-rc",type=float,default=1.,help="Characteristic Radius")
parser.add_argument("-q",type=float,default=0.8,help="Flattening Parameter")

parser.add_argument("-Emin",type=float,default=20)
parser.add_argument("-Emax",type=float,default=200)
parser.add_argument("-nb_E",type=int,default=5,help="Number of energy slices in tomographic mode")

parser.add_argument("-open",type=str,default=None)
args = parser.parse_args()

tf = args.tf
t_span = (0,tf)
nb_points_orbit = 20000
t_eval = np.linspace(0,tf,nb_points_orbit) # ts at which the output of the integrator is stored.
                                           # in reality, integration normally stops before tf
                                           # (if a the given number of crossings is reached)
                                           # and so a large part of these ts will not be used.
# Decide whether to terminate integration after N crossings or integrate until tf
def event_yplanecross(t,y):
    return y[1]
event_yplanecross.direction = 1
event_count_max = args.nb_crossings

def vy(x,vx):
    ED = 2*(args.E-pot.potential([x,0])) - vx**2
    if ED < 0:
        return None
    else:
        return np.sqrt(ED)

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

fig, ax = plt.subplots(1,2,figsize=fs)
ax[0].set_xlabel('$x$ [kpc]')
ax[0].set_ylabel('$v_x$ [km/s]')
ax[1].set_xlabel('$x$ [kpc]')
ax[1].set_ylabel('$y$ [kpc]')
ax[0].set_title("Poincaré section $y=0,v_y>0$")
ax[1].set_title("Orbit in $xy$-plane")

lines_sections = [ax[0].plot([], [],'o',ms=0.3,color='black',picker=True)[0] for i in range(args.nb_orbs)]
line_orbit = ax[1].plot([], [],lw=1)[0]
E_range = np.linspace(args.Emin,args.Emax,args.nb_E)
ensemblelist = []
is_new = args.open is None
print("Generating plots for {:.0f} energy levels...".format(args.nb_E))
for e in tqdm(E_range):
    ensemblelist.append(OrbitEnsemble(pot,lines_sections,line_orbit,e,is_new))

tom = TomographicPlot(fig,ensemblelist)

plt.show()