import numpy as np
from scipy.integrate._ivp.rk import DOP853
from scipy.integrate._ivp.ivp import prepare_events, handle_events, find_active_events, OdeResult
from scipy.optimize import OptimizeResult

'''
Custom wrapper around scipy's DOP853 integrator, largely inspired by the solve_ivp method
but containing some additional features and discarding functionalities not required by this program.

DISCLAIMER: Some portions of the code are taken litteraly from scipy's source,
            specifically scipy/integrate/ivp.py).

Newly added:
The user can either specify a time span (ti,tf) for the integration OR specify a number of times a
given event (typically a plane crossing) must happen for the integration to stop.

Please Note:
The original solve_ivp can deal with multiple event types and outputs ys and ts for each type. The
counting function implemented here assumes that only a single event is provided. If this is not
the case, expect false behaviour. Nonetheless, since the event watching code is copied from the
original solve_ivp, it still handles events as if multiple could be provided.
TODO: either rewrite to make it explicit that only 1 event is watched OR implement a counting
      that also works for multiple events.

Parameters
---------
fun: callable
...
'''

# The following function is a remade version of solve_ivp, which reuses some parts of the original
def integrate_orbit(fun, t_span, y0, t_eval=None, events=None,event_count_end = None, **options):
    t0, tf = map(float,t_span)
    solver = DOP853(fun,t0,y0,tf,**options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    else:
        ts = []
        ys = []

    t_eval_i = 0

    events, is_terminal, event_dir = prepare_events(events)

    if events is not None:
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    event_count = 0

    while status is None:
        message = solver.step()
        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y
        sol = None

        if events is not None:
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, is_terminal, t_old, t)
                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                # NEW: Stop the integration after event_count_max events have occured
                if event_count_end is not None:
                    event_count += 1
                    if event_count >= event_count_end:
                        terminate = True

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        if t_eval is None:
            ts.append(t)
            ys.append(y)
        else:
            t_eval_i_new = np.searchsorted(t_eval, t, side='right')
            t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

    if event_count_end is not None and status == 0:
        print("Warning: the time limit tf={:.1f} was reached before {:.0f} crossings could occur".format(tf,event_count_end))
    
    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    return OdeResult(t=ts,y=ys,t_events = t_events, y_events = y_events,status=status)