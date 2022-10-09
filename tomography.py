import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import argparse
import solver
import pickle as pkl

fs = (15,7)

parser = argparse.ArgumentParser(
    description="Poincare x vx Section of Logarithmic potential"
)

parser.add_argument("-tf",type=float,default=100,help="Maximal integration time. If --no_count, this will be the obligatory final time")
parser.add_argument("-nb_crossings",type=int,default= 40,help="Terminate integration after n crossings of the plane (=nb of points in the Poincaré map)")
parser.add_argument("-nb_orbs",type=int,default=11,help="Number of orbits to sample if --fill is passed")
parser.add_argument("--no_count",action='store_true',help="Integrate until t=tf, without counting the crossings")


parser.add_argument("-v0",type=float,default=10.,help="Characteristic Velocity")
parser.add_argument("-rc",type=float,default=1.,help="Characteristic Radius")
parser.add_argument("-q",type=float,default=0.8,help="Flattening Parameter")

parser.add_argument("-Emin",type=float,default=20)
parser.add_argument("-Emax",type=float,default=200)
parser.add_argument("-nb_E",type=int,default=5,help="Number of energy slices in tomographic mode")

parser.add_argument("--progress",action='store_true',help="Use tqdm to show progress bar")

parser.add_argument("--save",action='store_true')
parser.add_argument("-open",type=str,default=None)
args = parser.parse_args()

"""
Reorganized version of logpoincare for tomographic plots.
Objective: separate data generation and visualisation to be able to export data

"""

# Integration Parameters
t_span = (0,args.tf)
nb_points_orbit = 20000
t_eval = np.linspace(0,args.tf,nb_points_orbit) # ts at which the output of the integrator is stored.
                                           # in reality, integration normally stops before tf
                                           # (if a the given number of crossings is reached)
                                           # and so a large part of these ts will not be used.

# Decide whether to terminate integration after N crossings or integrate until tf
if args.no_count:
    event_count_max = None
else:
    event_count_max = args.nb_crossings

# Event function (y plane crossing)
def event_yplanecross(t,y):
    return y[1]
event_yplanecross.direction = 1

# Progress bar:
if args.progress:
    from tqdm import tqdm
    progbar = lambda itb: tqdm(itb)
else:
    progbar = lambda itb: itb
class Potential:
    def __init__(self):
        pass
    
    def phi(self):
        pass
    def accel(self):
        pass
    def RHS(self,t,y):
        F = self.accel(y)
        return [y[2],
                y[3],
                F[0],
                F[1]]
    def info(self):
        s = "Empty Potential"
        return(s)
class LogarithmicPotential(Potential):
    def __init__(self,v0,rc,q):
        self.v0 = v0
        self.rc = rc
        self.q = q
    def phi(self,y):
        return 0.5*self.v0**2 * np.log(self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
    def accel(self,y):
        A = - self.v0**2 / (self.rc**2 + y[0]**2 + y[1]**2/self.q**2)
        ax = A*y[0]
        ay = A*y[1]/(self.q**2)
        return np.array([ax,ay])
    def maxval_x(self,E):
        return np.sqrt(np.exp(2*E/self.v0**2)-self.rc**2)
    def ydot(self,E,x,xdot):
        ED = 2*(E-self.phi([x,0])) - xdot**2
        if ED < 0:
            return None
        else:
            return np.sqrt(ED)
    def info(self):
        return("Logarithmic potential: v0 = {:.1f}, rc = {:.1f}, q  = {:.1f}".format(self.v0,self.rc,self.q))
    
class PoincareSection:
    def __init__(self,nb_orbits,nb_points) -> None:
        self.sections = np.empty((nb_orbits,nb_points,2))#[] # Array of shape ()

class PoincareCollection:
    def __init__(self,E_list,orbits_list,sections_list,potential_info="No information about potential") -> None:
        if not (len(E_list) == len(orbits_list) == len(sections_list)):
            raise ValueError("lists must be of the same length")
        self.energylist = E_list
        self.orbitslist = orbits_list
        self.sectionslist = sections_list
        self._potentialinfo = potential_info
        self.nb_energies = len(E_list)
        self.nb_trajectories = len(orbits_list[0])
    def potential_info(self):
        print(self._potentialinfo)

def integrate_energy(pot,E,N_orbits):
    xlim = 0.9999*pot.maxval_x(E)
    x_ic = np.linspace(-xlim,xlim,N_orbits)
    ydot_ic = np.sqrt(2*(E-pot.phi([x_ic,np.zeros(N_orbits)])))
    f = lambda t,y: pot.RHS(t,y)
    orbits = []
    sections = []
    for k in range(N_orbits):
        y0 = [x_ic[k],0,0,ydot_ic[k]]
        res = solver.integrate_orbit(f,t_span,y0,t_eval=t_eval,events=event_yplanecross,event_count_end=event_count_max)
        orbits.append(res['y'][0:2])
        sections.append(res['y_events'][0][:,[0,2]].T)
    return orbits,sections

class Tomography:
    def __init__(self,ax_sec: Axes,ax_orb: Axes,data: PoincareCollection) -> None:
        self.collection = data
        self.lines_sec = [ax_sec.plot([], [],'o',ms=0.3,color='black',picker=True)[0] for i in range(data.nb_trajectories)]
        self.lines_orb = ax_orb.plot([], [],lw=1)[0]
        self.idx = 0
        self.fig = ax_sec.figure
        self.fig.canvas.mpl_connect('key_press_event',self)
        self.show(0)
    def __call__(self,event):
        if event.key == 'up':
            ii = self.idx + 1
        elif event.key == 'down':
            ii = self.idx - 1
        if ii in range(self.collection.nb_energies):
            self.show(ii)
            #self.ensemblelist[self.idx].hide()
            #self.ensemblelist[ii].show()
            self.idx = ii
    def show(self,idx):
        for k,l in enumerate(self.lines_sec):
            #print(self.collection.sectionslist[idx][k][0])
            l.set_xdata(self.collection.sectionslist[idx][k][0])
            l.set_ydata(self.collection.sectionslist[idx][k][1])
        #self.lines_orb.set_xdata(self.collection.orbitslist[idx][1][0])
        #self.lines_orb.set_ydata(self.collection.orbitslist[idx][1][1])
        #otest = self.collection.orbitslist[idx][1]
        #self.lines_orb.axes.plot(otest[0],otest[1])
        #self.lines_orb.set_xdata(otest[0])
        #self.lines_orb.set_ydata(otest[1])
        self.lines_sec[0].axes.relim()
        self.lines_sec[0].axes.autoscale()
        self.lines_sec[0].axes.figure.canvas.draw()
        #self.lines_orb.axes.relim()
        #self.lines_orb.axes.autoscale()
        #self.lines_orb.axes.figure.canvas.draw()
if __name__ == "__main__":
    if args.open is None:
        pot = LogarithmicPotential(args.v0,args.rc,args.q)
        E_range = np.linspace(args.Emin,args.Emax,args.nb_E)
        orbslist = []
        secslist = []
        for e in progbar(E_range):
            o,s = integrate_energy(pot,e,args.nb_orbs)
            orbslist.append(o)
            secslist.append(s)
        
        col = PoincareCollection(E_range,orbslist,secslist,pot.info())
    else:
        with open(args.open,'rb') as f:
            col = pkl.load(f)


    fig, ax = plt.subplots(1,2,figsize=fs)
    tom = Tomography(ax[0],ax[1],col)
    col.potential_info()
    if args.save:
        with open('PoincareCollection.pkl','wb') as f:
            pkl.dump(col,f)
    plt.show()
    #otest = col.orbitslist[1][2]
    #plt.plot(otest[0],otest[1])
    #plt.show()
    #    pass
    #orb, sec = integrate_energy(pot,args.Emin,5)
"""
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
"""