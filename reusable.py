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