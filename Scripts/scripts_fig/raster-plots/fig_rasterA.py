from preamble import *


params = [
    (0.04, 0.2, 3, 1),
    (0.2,  0.2, 3, 1),
    (0.6,  0.2, 3, 1),
]
binsizes_ms = [50, 5, 5]
ms = [4, 3, 3]

num_cols = len(params)
fig, axes = plt.subplots(ncols=num_cols, figsize=[12,4])
[ax.tick_params(labelleft=False) for ax in axes]

for i, (gE,gI,alpha,kappa) in enumerate(params):
    nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    qgraph.raster_plot(nd.dynamics.spike_times[::20], ax=axes[i], colors=["navy" for _ in range(10)]+["orangered" for _ in range(40)], marker=".", mec="none", ms=ms[i])

[ax.set(ylim=(0.5,50.5), yticks=[]) for ax in axes]
[ax.tick_params(axis="x", which="both", length=0) for ax in axes]
axes[0].set(xlim=(1,40), xticks=[1,10,20,30,40])
axes[1].set(xlim=(1,3))
axes[2].set(xlim=(1,1.8))
axes[0].set(ylabel="50 neurons")
[ax.set(xlabel="time $t$ (s)") for ax in axes]

fig.tight_layout()
fig.subplots_adjust(hspace=.05, wspace=.3)
# plt.show()
fig.savefig("fig_rasterA.pdf", dpi=400)
fig.savefig("fig_rasterA.png", dpi=400)