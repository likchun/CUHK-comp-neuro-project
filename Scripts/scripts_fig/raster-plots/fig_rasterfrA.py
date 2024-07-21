from preamble import *


params = [
    (0.04, 0.2, 3, 1),
    (0.2,  0.2, 3, 1),
    (0.6,  0.2, 3, 1),
]
binsizes_ms = [50, 5, 5]
ms = [4, 3, 3]

num_cols = len(params)
fig, axes = plt.subplots(nrows=2, ncols=num_cols, gridspec_kw={"height_ratios":[1,.4]}, figsize=[10,5])
[axes[i][j].tick_params(labelbottom=False) for i in np.arange(5)[[0]] for j in range(3)]
[ax.tick_params(labelleft=False) for ax in axes[0]]

for i, (gE,gI,alpha,kappa) in enumerate(params):
    nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    qgraph.raster_plot(nd.dynamics.spike_times[::20], ax=axes[0][i], colors=["navy" for _ in range(10)]+["orangered" for _ in range(40)], marker=".", mec="none", ms=ms[i])
    axes[1][i].plot(*nd.dynamics.average_firing_rate_time_histogram(binsize_ms=binsizes_ms[i]), c="k", lw=1)

[ax.set(ylim=(0.5,50.5), yticks=[]) for ax in axes[0]]
[ax.tick_params(axis="x", which="both", length=0) for ax in axes[0]]
[ax.set(xlim=(1,20),  xticks=[1,10,20]) for ax in axes.T[0]]
[ax.set(xlim=(1,2),   xticks=[1,1.5,2],   xticklabels=["1","1.5","2"]) for ax in axes.T[1]]
[ax.set(xlim=(1,1.4), xticks=[1,1.2,1.4], xticklabels=["1","1.2","1.4"]) for ax in axes.T[2]]
[ax.set(ylim=(None,None), xlabel="$t$ (s)") for ax in axes[1]]
axes[1][0].set(ylim=(-2*.05, 2),     yticks=[0, 1, 2])
axes[1][1].set(ylim=(-800*.05, 800), yticks=[0, 400, 800])
axes[1][2].set(ylim=(-600*.05, 600), yticks=[0, 300, 600])
axes[0][0].set(ylabel="50 neurons")
axes[1][0].set(ylabel="$r(t)$ (Hz)")

axes[0][0].text(0,1.1, "(a): I", transform=axes[0][0].transAxes)
axes[0][1].text(0,1.1, "(b): II", transform=axes[0][1].transAxes)
axes[0][2].text(0,1.1, "(c): III", transform=axes[0][2].transAxes)

fig.tight_layout()
fig.subplots_adjust(hspace=.05, wspace=.3)
# plt.show()
fig.savefig("fig_rasterfrA.pdf", dpi=400)
fig.savefig("fig_rasterfrA.png", dpi=400)
