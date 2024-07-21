from preamble import *


fig, [ax1,ax2] = plt.subplots(1,2,figsize=[10,5])

m1, = ax1.plot([],[], "k.", ms=8)
l1, = ax1.plot([],[], "r-")
m2, = ax1.plot([],[], "kx", ms=8)
l2, = ax1.plot([],[], "r--")
ax1.legend([(l1, m1), (l2, m2)], ["2.5", "3.0"], title=r"$\alpha$")

nd = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, "2.5"))
nd.remove_dynamics(500,0)
x, y = qgraph.prob_dens_plot(np.hstack(nd.dynamics.interspike_intervals), 1.4, ax=ax1, c="k", ls="", marker=".", ms=4, label="2.5")
fit_range = (.035 < x) & (x < 150)
coef = np.polyfit(x[fit_range], np.log10(y[fit_range]), 1)
ax1.plot(x, np.power(10, np.poly1d(coef)(x)), "r-")
print("fit exponent: {:5f}".format(np.power(10,coef[1])))
print("firing rate: {:5f}".format(nd.dynamics.mean_firing_rate.mean()))

nd = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, "3.0"))
nd.remove_dynamics(500,0)
x, y = qgraph.prob_dens_plot(np.hstack(nd.dynamics.interspike_intervals), .4, ax=ax1, c="k", ls="", marker="x", ms=4, label="3.0")
fit_range = (.35 < x) & (x < 15)
coef = np.polyfit(x[fit_range], np.log10(y[fit_range]), 1)
ax1.plot(x, np.power(10, np.poly1d(coef)(x)), "r--")
print("fit exponent: {:5f}".format(np.power(10,coef[1])))
print("firing rate: {:5f}".format(nd.dynamics.mean_firing_rate.mean()))

ax1.set(xlim=(0, 300), xticks=[0,50,100,150,200,250,300], ylim=(1e-5, 1e0))
ax1.set(yscale="log")
ax1.set(xlabel="ISI (s)", ylabel="$P$(ISI)")

ax2.plot([],[], "k.", ms=8, label="5.0")
ax2.plot([],[], "kx", ms=8, label="8.0")
ax2.legend(title=r"$\alpha$")

nd = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, "5.0"))
nd.remove_dynamics(500,0)
x, y = qgraph.prob_dens_plot(np.hstack(nd.dynamics.interspike_intervals), .008, ax=ax2, c="k", ls="", marker=".", ms=4)

nd = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, "8.0"))
nd.remove_dynamics(500,0)
x, y = qgraph.prob_dens_plot(np.hstack(nd.dynamics.interspike_intervals), .004, ax=ax2, c="k", ls="", marker="x", ms=4)

ax2.set(xlim=(0, 1), ylim=(1e-3, 1e2))
ax2.set(yscale="log")
ax2.set(xlabel="ISI (s)")

fig.tight_layout()
# plt.show()
fig.savefig("fig_ISIexc.pdf", dpi=400)
fig.savefig("fig_ISIexc.png", dpi=400)