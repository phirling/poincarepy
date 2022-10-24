def integrate_energy(pot,E,N_orbits,t_span,t_eval,event,event_count_max,xlim=None):
    if xlim is None:
        g = lambda x: E-pot.phi(np.array([x,np.zeros_like(x)]))
        gprime = lambda x: pot.accel(np.array([x,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x)]))[0]
        xlim = 0.999*scpopt.newton(g,(-1,1),gprime)
        print(xlim)
    x_ic = np.linspace(xlim[0],xlim[1],N_orbits)
    ydot_ic = np.sqrt(2*(E-pot.phi([x_ic,np.zeros(N_orbits)])))

    f = lambda t,y: pot.RHS(t,y)
    orbits = []
    sections = []
    for k in range(N_orbits):
        y0 = [x_ic[k],0,0,ydot_ic[k]]
        res = solver.integrate_orbit(f,t_span,y0,t_eval=t_eval,events=event,event_count_end=event_count_max)
        orbits.append(res['y'][0:2])
        sections.append(res['y_events'][0][:,[0,2]].T)
    return orbits,sections

     ## WIP
    def _to_bdry(self,q,E,xlim,origin=(0.,0.)):
        x0,y0 = origin[0],origin[1]
        x1,y1 = q[0],q[1]
        a,b = xlim[0],xlim[1]
        if x1 < x0:
            a = max(a,x1)
            b = min(b,x0)
        else:
            a = max(a,x0)
            b = min(b,x1)

        print(a,b)
        f = lambda x: np.sign(y1)*self.vxlim(E,x) - (y0*(x1-x) + y1*(x-x0))/(x1-x0)
        mlt = 5
        root = scpopt.brentq(f,a,b)
        xb = root - mlt*np.sign(x1)*self._dx
        yb = np.sign(y1)*self.vxlim(E,root) - mlt*np.sign(y1)*self._dvx
        return xb,yb
    def _manage_bdry(self,q,E,xlim,xtol=2e-3,ytol=2e-3):
        x, y = q[0], q[1]
        if q[0] < xlim[0] + xtol: x += xtol
        elif q[0] > xlim[1] - xtol: x -= xtol
        #zvc = self.vxlim(E,x)
        #if q[1] < zvc + ytol: y += ytol
        #elif q[1] > zvc - ytol: y -= ytol
        return x,y