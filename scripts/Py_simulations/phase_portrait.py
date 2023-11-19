import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

plt.rc("font", family="courier", size=20)


"""Settings"""
# templates after code

# I = 3.79
I = 0
a = 0.02
b = 0.2
c = -50
d = 2
init_point  = {"v": -62.5, "u": -12.5}
vfield_range = {"v": (-80, -45), "u": (-15, -5)}


duration = 900 # in how many steps will the trajectory terminate
ode_stepsize = 0.005

v_peak = 30




# I = lambda t: ...
dv = lambda x, t: 0.04*x[0]*x[0] + 5*x[0] + 140 - x[1] + I
du = lambda x: a*(b*x[0] - x[1])
def vField_izhikevich(t, x): return [dv(x, t), du(x)] # Define vector field

def v_nullcline(vfield_range): # dv/dt = 0
    v = np.arange(vfield_range["v"][0]*100, vfield_range["v"][1]*100) / 100
    u = 0.04*v*v + 5*v + 140 + I
    return np.vstack((v, u))

def u_nullcline(vfield_range):
    u = np.arange(vfield_range["u"][0]*100, vfield_range["u"][1]*100) / 100
    v = u / b
    return np.vstack((v, u))

def fixed_points():
    det = (b-5)**2-4*0.04*(I+140)
    if det > 0:
        v1, v2 = (b-5 + np.sqrt(det))/(2*0.04), (b-5 - np.sqrt(det))/(2*0.04)
        u1, u2 = b*v1, b*v2
        fixpt = [(v1, u1), (v2, u2)]
    elif det == 0:
        v1 = (b-5)/(2*0.04)
        u1 = b*v1
        fixpt = [(v1, u1)]
    else: fixpt = []
    return fixpt

def show_background_color():
    _V, _U = np.mgrid[vfield_range["v"][0]:29.1:71j,vfield_range["u"][0]:vfield_range["u"][1]:65j]
    _dV, _dU = vField_izhikevich(0, [_V, _U])
    regionA = np.vstack((_V[(_dV > 0) & (_dU > 0)], _U[(_dV > 0) & (_dU > 0)]))
    regionB = np.vstack((_V[(_dV > 0) & (_dU < 0)], _U[(_dV > 0) & (_dU < 0)]))
    regionC = np.vstack((_V[(_dV < 0) & (_dU > 0)], _U[(_dV < 0) & (_dU > 0)]))
    regionD = np.vstack((_V[(_dV < 0) & (_dU < 0)], _U[(_dV < 0) & (_dU < 0)]))
    ax.scatter(*regionA, c="maroon",    alpha=.1, marker='s', edgecolors="none")#, label="dv/dt > 0 & du/dt > 0")
    ax.scatter(*regionB, c="peru",      alpha=.1, marker='s', edgecolors="none")#, label="dv/dt > 0 & du/dt < 0")
    ax.scatter(*regionC, c="darkgreen", alpha=.1, marker='s', edgecolors="none")#, label="dv/dt < 0 & du/dt > 0")
    ax.scatter(*regionD, c="navy",      alpha=.1, marker='s', edgecolors="none")#, label="dv/dt < 0 & du/dt < 0")


fig, ax = plt.subplots(figsize=(9, 7))

show_background_color()

V, U = np.mgrid[vfield_range["v"][0]:vfield_range["v"][1]:20j,vfield_range["u"][0]:vfield_range["u"][1]:30j]
dV, dU = vField_izhikevich(0, [V, U])
# vector field
ax.quiver(V, U, dV, dU, alpha=.4, zorder=1)
# nullclines
ax.plot(*v_nullcline(vfield_range), "--", c="m", lw=2, label="v nullcline", zorder=2)
ax.plot(*u_nullcline(vfield_range), "--", c="g", lw=2, label="u nullcline", zorder=2)
# spike cutoff and reset
ax.plot([c, c], [vfield_range["u"][0], vfield_range["u"][1]], '-.', c="saddlebrown", lw=1.5, zorder=3)
ax.plot([v_peak, v_peak], [vfield_range["u"][0], vfield_range["u"][1]], '-.', c="saddlebrown", lw=1.5, zorder=3)
# fixed points
for xy in fixed_points():
    ax.scatter(*xy, s=40, c="none", marker='s', edgecolor='k', linewidths=1.5, zorder=4)
ax.scatter([],[], c="none", marker='s', edgecolor='k', linewidths=1.5, label="fixed point")
# initial point
ic = (init_point["v"], init_point["u"])
ax.scatter(ic[0], ic[1], c='b', label="initial val", zorder=6)
# trajectory
ode = ode(vField_izhikevich)
ode.set_integrator("vode", nsteps=500, method="bdf") # BDF method suited to stiff systems of ODEs
ode.set_initial_value(ic, t=0)
ts, vu = [], []
resets = [0]
num_spk = 0
step = 0
while ode.successful() and (ode.t < duration):
    if ode.y[0] > v_peak:
        resets.append(step)
        ode.set_initial_value((c, ode.y[1]+d), t=ode.t)
    step += 1
    ts.append(ode.t)
    vu.append(ode.y)
    ode.integrate(ode.t+ode_stepsize)
print("TOTAL STEPS: {}".format(step))
resets.append(step)
ts, vu = np.array(ts), np.array(vu)
num_spk = len(resets)-2
if num_spk < 1:
    ax.plot(*vu.T, color='r', zorder=5)
else:
    for r in range(len(resets))[1:]:
        ax.plot(*vu[resets[r-1]:resets[r]].T, 'r', alpha=1, lw=2, zorder=5)
# legend and text
ax.text(c+3, vfield_range["u"][1]*1.05, "spike\nreset",
        fontdict=dict(fontsize=13, c="saddlebrown"), verticalalignment="top",
        bbox=dict(facecolor="w", edgecolor="saddlebrown", alpha=.85), zorder=3)
ax.text(v_peak-3, vfield_range["u"][1]*1.05, "spike\ncutoff",
        fontdict=dict(fontsize=13, c="saddlebrown"), verticalalignment="top", horizontalalignment="right",
        bbox=dict(facecolor="w", edgecolor="saddlebrown", alpha=.85), zorder=3)
str_param = "Parameters:\na = {0}\nb = {1}\nc = {2}\nd = {3}\nI$_{{\\rm ext}}$ = {4}".format(a, b, c, d, I)
if len(fixed_points()) == 1:
    str_fixpts = "Fixed pt:\n{0:.1f},{1:.1f}".format(*fixed_points()[0])
elif len(fixed_points()) == 2:
    str_fixpts = "Fixed pts:\n{0:.1f},{1:.1f}\n{2:.1f},{3:.1f}".format(*fixed_points()[0], *fixed_points()[-1])
else:
    str_fixpts = "Fixed pt:\nnone"
str_initpt = "Initial:\nv$_0$ = {0}\nu$_0$ = {1}".format(*ic)
ax.text(v_peak+7.5, (vfield_range["u"][0]+vfield_range["u"][1])/2,
        str_param+"\n-----------\n"+str_fixpts+"\n-----------\n"+str_initpt,
        fontdict=dict(fontsize=13, c="k"), linespacing=2, verticalalignment="center", horizontalalignment="left",
        bbox=dict(boxstyle="square,pad=.75", facecolor="w", edgecolor="k", alpha=1), zorder=3)
ax.legend(fontsize=13, loc='lower right', edgecolor='k', fancybox=False, shadow=False, ncol=1)
ax.set_title("Phase portrait of\nIzhikevich's neuron spiking model", fontdict=dict(fontsize=15), loc="left")
ax.set(xlabel="Membrane potential v (mV)", ylabel="Recovery variable u")
ax.set(xlim=(vfield_range["v"][0], v_peak+3))
ax.set(ylim=(vfield_range["u"][0], vfield_range["u"][1]))

# info
print("NUMBER OF SPIKES: {:d}".format(num_spk))
print("FIXED POINTS:")
print(fixed_points())

plt.tight_layout()
plt.show()
# fig.savefig("phase portrait.png")





exit(0)

templates

1.
one_spike
I = 3.77
a = 0.02
b = 0.2
c = -65
d = 2
init_point  = {"v": -70, "u": -14}
vfield_range = {"v": (-75, -40), "u": (-15, -10)}

2.
bursting
I = 3.79
a = 0.02
b = 0.2
c = -50
d = 2
init_point  = {"v": -62.5, "u": -12.5}
vfield_range = {"v": (-80, -45), "u": (-15, -5)}

3.
I = 3.78
a = 0.02
b = 0.2
c = -50
d = 2
init_point  = {"v": -62.5, "u": -12.5}
vfield_range = {"v": (-64, -61), "u": (-12.8, -12.1)}

4.
I = 10
I = 4
a = 0.02
b = 0.2
c = -50
d = 2
init_point  = {"v": -62.25, "u": -12.45}
vfield_range = {"v": (-80, -45), "u": (-14.5, -4.5)}