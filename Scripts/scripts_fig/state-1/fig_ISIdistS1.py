from preamble import *
from matplotlib import ticker


fig, [ax1,ax2] = plt.subplots(1, 2, figsize=[8,4.5], sharex=True, sharey=True)


params = [
    # (0,0.2),
    (0.01,0.2),
    # (0.02,0.2),
    # (0.03,0.2),
    (0.04,0.2),
    (0.05,0.2),
]

nds = [NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},3,1".format(gE,gI))) for gE,gI in params]

c = ["r","r","r"]
m = [".","x","+"]
ls = ["-","--",":"]
ms = [4,4,5]

m1, = ax1.plot([],[], "k.", ms=8)
l1, = ax1.plot([],[], "r-")
m2, = ax1.plot([],[], "kx", ms=8)
l2, = ax1.plot([],[], "r--")
m3, = ax1.plot([],[], "k+", ms=10)
l3, = ax1.plot([],[], "r:")
ax1.legend([(l1, m1), (l2, m2), (l3, m3)], ["0.01","0.04","0.05"], title=r"$g_E$")

for i,nd in enumerate(nds):
    x, y = qgraph.prob_dens_plot(np.hstack(nd.dynamics.interspike_intervals[200:1000]), .2, ax=ax1, c="k", ls="", marker=m[i], ms=ms[i])

    if i==0:   fit_range = (.35 < x) & (x < 18)
    elif i==1: fit_range = (.35 < x) & (x < 11)
    elif i==2: fit_range = (.35 < x) & (x < 6)
    coef = np.polyfit(x[fit_range], np.log10(y[fit_range]), 1)
    ax1.plot(x, np.power(10, np.poly1d(coef)(x)), ls=ls[i], c=c[i])
    print(coef)

ax1.set(xlim=(0, 30), ylim=(1e-4, 1e0))
ax1.set(yscale="log")
ax1.set(xlabel="ISI (s)", ylabel="$P$(ISI)")
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))


params = [
    (0.04,0),
    (0.04,0.2),
    (0.04,0.6),
    (0.04,8),
]

nds = [NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},3,1".format(gE,gI))) for gE,gI in params]

c = ["r","r","r","r"]
m = [".","x","+","*"]
ls = ["-","-.","--",":"]
ms = [4,4,4,4]

m1, = ax2.plot([],[], "k.", ms=8)
l1, = ax2.plot([],[], "r-")
m2, = ax2.plot([],[], "kx", ms=8)
l2, = ax2.plot([],[], "r-.")
m3, = ax2.plot([],[], "k+", ms=10)
l3, = ax2.plot([],[], "r--")
m4, = ax2.plot([],[], "k*", ms=8)
l4, = ax2.plot([],[], "r:")
ax2.legend([(l1, m1), (l2, m2), (l3, m3), (l4, m4)], ["0.0","0.2","0.6","8.0"], title=r"$g_I$")

for i,nd in enumerate(nds):
    x, y = qgraph.prob_dens_plot(np.hstack(nd.dynamics.interspike_intervals[200:1000]), .2, ax=ax2, c="k", ls="", marker=m[i], ms=ms[i])

    fit_range = (.35 < x) & (x < 11)
    coef = np.polyfit(x[fit_range], np.log10(y[fit_range]), 1)
    ax2.plot(x, np.power(10, np.poly1d(coef)(x)), ls=ls[i], c=c[i])
    print(coef)

ax2.set(xlim=(0, 30), ylim=(1e-4, 1e0))
ax2.set(yscale="log")
ax2.set(xlabel="ISI (s)")
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax1.text(0,1.1, "(a)", transform=ax1.transAxes)
ax2.text(0,1.1, "(b)", transform=ax2.transAxes)

fig.tight_layout()
# plt.show()
fig.savefig("fig_ISIdistS1.pdf", dpi=400)
fig.savefig("fig_ISIdistS1.png", dpi=400)