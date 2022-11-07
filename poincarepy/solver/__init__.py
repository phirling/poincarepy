import numpy as np
from scipy.integrate._ivp.rk import DOP853
from scipy.integrate._ivp.ivp import prepare_events, handle_events, find_active_events, OdeResult, OdeSolution

'''
Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modified version of SciPy's solve_ivp (scipy/integrate/ivp.py), to be able to terminate integration after a given
number of events occured.

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
'''

# The following function is a remade version of solve_ivp, which reuses some parts of the original
def integrate_orbit(fun, t_span, y0, t_eval=None, events=None,event_count_end = None,dense_output=False, **options):
    t0, tf = map(float,t_span)
    solver = DOP853(fun,t0,y0,tf,**options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    t_eval_i = 0
    interpolants = []

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


        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
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
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    if event_count_end is not None and status == 0:
        #print("Warning: the time limit tf={:.1f} was reached before {:.0f} crossings could occur".format(tf,event_count_end))
        raise RuntimeError("The time limit tf={:.1f} was reached before {:.0f} crossings could occur".format(tf,event_count_end))
    
    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(ts, interpolants)
        else:
            sol = OdeSolution(ti, interpolants)
    else:
        sol = None

    return OdeResult(t=ts,y=ys,sol=sol,t_events = t_events, y_events = y_events,status=status)