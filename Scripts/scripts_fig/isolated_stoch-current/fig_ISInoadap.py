from preamble import *


nd1 = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, "8.0"))
nd0 = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, "8.0_noadap"))
nd0.remove_dynamics(500,0)
nd1.remove_dynamics(500,0)

fig, ax = plt.subplots(figsize=[6,5])

x, y = qgraph.prob_dens_plot(np.hstack(nd1.dynamics.interspike_intervals), .001, ax=ax, c="k", ls="-", marker="x", lw=0, ms=4)

x, y = qgraph.prob_dens_plot(np.hstack(nd0.dynamics.interspike_intervals), .001, ax=ax, c="k", ls="-", marker=".", lw=0, ms=4)
fit_range = np.isfinite(np.log10(y))
coef, residue, _,_,_ = np.polyfit(x[fit_range], np.log10(y[fit_range]), 1, full=True)
ax.plot(x, np.power(10, np.poly1d(coef)(x)), "r--")

isi = np.hstack(nd0.dynamics.interspike_intervals)
print("Cv: {:5f}".format(np.std(isi)/np.mean(isi)))

print(coef)
print("fit exponent: {:5f}".format(np.power(10,coef[1])))
print("firing rate: {:5f}".format(nd0.dynamics.mean_firing_rate.mean()))
print("error sum of squares: {:5f}".format(residue[0]))

ax.set(xlim=(0,.35), xticks=[0,.05,.1,.15,.2,.25,.3,.35], ylim=(1e-3, 1e2))
ax.set(yscale="log")
ax.set(xlabel="ISI (s)", ylabel="$P$(ISI)")
fig.tight_layout()
# plt.show()
fig.savefig("fig_ISInoadap.pdf", dpi=400)
fig.savefig("fig_ISInoadap.png", dpi=400)