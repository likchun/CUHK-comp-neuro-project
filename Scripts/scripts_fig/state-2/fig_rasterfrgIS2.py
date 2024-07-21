from preamble import *


params = [
    (0.2, 0.2, 3, 1),
    (0.2, 0.6, 3, 1),
    (0.2, 8,   3, 1),
]
binsizes_ms = [5, 5, 5]
ms = [3, 3, 3]

num_cols = len(params)
fig, axes = plt.subplots(nrows=2, ncols=num_cols, gridspec_kw={"height_ratios":[1.2,1]}, figsize=[10,5])
[axes[i][j].tick_params(labelbottom=False) for i in np.arange(5)[[0]] for j in range(3)]
[ax.tick_params(labelleft=False) for ax in axes[0]]

for i, (gE,gI,alpha,kappa) in enumerate(params):
    nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    qgraph.raster_plot(nd.dynamics.spike_times[::20], ax=axes[0][i], colors=["navy" for _ in range(10)]+["orangered" for _ in range(40)], marker=".", mec="none", ms=ms[i])
    # qgraph.raster_plot(nd.dynamics.spike_times[::10], ax=axes[0][i], colors=["navy" for _ in range(20)]+["orangered" for _ in range(80)], marker=".", mec="none", ms=ms[i])
    axes[1][i].plot(*nd.dynamics.average_firing_rate_time_histogram(binsize_ms=binsizes_ms[i]), c="k", lw=1)

[ax.set(ylim=(0.5,50.5), yticks=[]) for ax in axes[0]]
[ax.tick_params(axis="x", which="both", length=0) for ax in axes[0]]
[[ax.set(xlim=(1,2), xticks=[1,1.2,1.4,1.6,1.8,2]) for ax in axes.T[i]] for i in range(3)]
[ax.set(ylim=(None,None), xlabel="$t$ (s)") for ax in axes[1]]
subfig_labels = ["(a)", "(b)", "(c)"]
# [ax.set_title("{}".format(subfig_labels[i]), fontsize=myFontSize1, pad=10) for i,ax in enumerate(axes[0])]
axes[1][0].set(ylim=(-800*.05, 800), yticks=[0, 200, 400, 600, 800])
axes[1][1].set(ylim=(-400*.05, 400), yticks=[0, 100, 200, 300, 400])
axes[1][2].set(ylim=(-160*.05, 160), yticks=[0, 40, 80, 120, 160])
axes[0][0].set(ylabel="50 neurons")
axes[1][0].set(ylabel="$r(t)$ (Hz)")

fig.tight_layout()
fig.subplots_adjust(hspace=.05, wspace=.25)
# plt.show()
fig.savefig("fig_rasterfrgIS2.pdf", dpi=400)
fig.savefig("fig_rasterfrgIS2.png", dpi=400)
