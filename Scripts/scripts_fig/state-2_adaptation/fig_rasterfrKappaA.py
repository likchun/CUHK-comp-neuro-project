from preamble import *


params = [
    # (0.2, 0.2, 3, 1),
    # (0.2, 0.2, 3, 0.3),
    # (0.2, 0.2, 3, 0.2),

    (0.2, 0.6, 3, 1),
    (0.2, 0.6, 3, 0.4),
    (0.2, 0.6, 3, 0.2),
]
binsizes_ms = [5, 5, 5]
ms = [3, 3, 3]

num_cols = len(params)
fig, axes = plt.subplots(nrows=2, ncols=num_cols, gridspec_kw={"height_ratios":[1,.4]}, figsize=[9,3.8], sharex=True)
[axes[i][j].tick_params(labelbottom=False) for i in np.arange(5)[[0]] for j in range(3)]
[ax.tick_params(labelleft=False) for ax in axes[0]]

for i, (gE,gI,alpha,kappa) in enumerate(params):
    try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    except FileNotFoundError: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    qgraph.raster_plot(nd.dynamics.spike_times[::20], ax=axes[0][i], colors=["navy" for _ in range(10)]+["orangered" for _ in range(40)], marker=".", mec="none", ms=ms[i])
    axes[1][i].plot(*nd.dynamics.average_firing_rate_time_histogram(binsize_ms=binsizes_ms[i]), c="k", lw=1)

[ax.set(xlim=(1,2), xticks=[1,1.5,2], xticklabels=["1","1.5","2"]) for ax in axes[0]]

[ax.set(ylim=(0.5,50.5), yticks=[]) for ax in axes[0]]
[ax.tick_params(axis="x", which="both", length=0) for ax in axes[0]]
axes[1][0].set(ylim=(-500*.05, 500), yticks=[0, 250, 500], xlabel="$t$ (s)")
[ax.set(ylim=(-500*.05, 500), yticks=[], xlabel="$t$ (s)") for ax in axes[1][1:]]
axes[0][0].set(ylabel="50 neurons")
axes[1][0].set(ylabel="$r(t)$")

axes[0][0].text(0,1.1, "(ai)", transform=axes[0][0].transAxes)
axes[0][1].text(0,1.1, "(aii)", transform=axes[0][1].transAxes)
axes[0][2].text(0,1.1, "(aiii)", transform=axes[0][2].transAxes)

fig.tight_layout()
fig.subplots_adjust(hspace=.05)
# plt.show()
fig.savefig("fig_rasterfrKappaA.pdf", dpi=400)
fig.savefig("fig_rasterfrKappaA.png", dpi=400)