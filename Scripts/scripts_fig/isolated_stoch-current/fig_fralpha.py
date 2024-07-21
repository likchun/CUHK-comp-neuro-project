from preamble import *


alphas = [
    "2.5", "2.8", "3.0", "3.5", "4.0", "5.0", "6.0", "7.0", "8.0"
]

fr_es, fr_is = [],[]
Cv_es, Cv_is = [],[]
for alpha in alphas:
    nd = NeuroData(os.path.join(data_path.SD_isolated_stoch_e, alpha))
    nd.remove_dynamics(500,0)
    fr_es.append(nd.dynamics.mean_firing_rate.mean())
    isi = np.hstack(nd.dynamics.interspike_intervals)
    Cv_es.append(np.std(isi)/np.mean(isi))
    nd = NeuroData(os.path.join(data_path.SD_isolated_stoch_i, alpha))
    nd.remove_dynamics(500,0)
    fr_is.append(nd.dynamics.mean_firing_rate.mean())
    isi = np.hstack(nd.dynamics.interspike_intervals)
    Cv_is.append(np.std(isi)/np.mean(isi))

fig, ax1 = plt.subplots(figsize=[6,5])
ax1.plot(np.array(alphas).astype(float), fr_es, "k.-",  lw=1)
ax1.plot(np.array(alphas).astype(float), fr_is, "k.--",  lw=1)
ax1.set(xlabel=r"$\alpha$", ylabel="$r_{0,E(I)}$ (Hz)")
ax1.set(xlim=(2.5,8), ylim=(0,35))
ax1.set(xticks=[2.5,3,4,5,6,7,8], xticklabels=["2.5","3","4","5","6","7","8"])

ax_in = ax1.inset_axes([.15, .44, .3, .5], transform=ax1.transAxes)
ax_in.plot(np.array(alphas).astype(float), fr_es, "k.-",  lw=1)
ax_in.plot(np.array(alphas).astype(float), fr_is, "k.--",  lw=1)
ax_in.set(xlim=(2.5,5), xticks=[2.5,3,4,5], xticklabels=["2.5","3","4","5"])
ax_in.set(yscale="log", ylim=(1e-3,1e1), yticks=[1e-3,1e-2,1e-1,1e0,1e1])

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(.1,.2,.3,.4,.5,.6,.7,.8,.9),numticks=12)
ax_in.yaxis.set_minor_locator(locmin)
ax_in.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

fig.tight_layout()
# plt.show()
fig.savefig("fig_fralpha.pdf")
fig.savefig("fig_fralpha.png")