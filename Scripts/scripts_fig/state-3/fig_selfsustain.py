from preamble import *


nds = [
    NeuroData("/Users/likchun/NeuroProject (CUHK MPhil)/Raw Data/network_A/transient_drive/0.6,0.2,3,1"),
    NeuroData("/Users/likchun/NeuroProject (CUHK MPhil)/Raw Data/network_A/transient_drive/0.2,0.2,3,0.1"),
    NeuroData("/Users/likchun/NeuroProject (CUHK MPhil)/Raw Data/network_A/transient_drive/0.2,0.2,3,1"),
]

fig, axes = plt.subplots(2, 3, sharex=True, sharey="row", gridspec_kw={"height_ratios":[1,.7]}, figsize=[10,5])
axes_twin = [ax.twinx() for ax in axes[1]]
skip = 20

[qgraph.raster_plot(nd.dynamics.spike_times[::skip], ax=ax, mec="none", ms=3, marker=".", time_scale="ms", colors=np.array(["navy" for _ in range(200)]+["orangered" for _ in range(800)])[::skip]) for nd,ax in zip(nds,axes[0])]
[ax.plot(*nd.dynamics.average_firing_rate_time_histogram(5, time_scale="ms"), "k-") for nd,ax in zip(nds,axes[1])]
[ax.plot(range(2500), np.concatenate([np.full(500,3),np.full(2000,0)]), "k--") for ax in axes_twin]

axes[0][0].set(ylabel="neuron", yticks=[])
[ax.set(
    xlim=(0, nds[0].configs["duration_ms"]),
    ylim=(.5, int(nds[0].configs["num_neuron"]/skip)+.5)
    ) for ax in axes[0]]
[ax.tick_params(axis="x", which="both",length=0) for ax in axes[0]]
axes[1][0].set(ylabel="$r(t)$ (Hz)")
[ax.set(xlabel="$t$ (ms)") for ax in axes[1]]
[ax.set(
    xlim=(100,1000), xticks=[100,500,1000],
    ylim=(800*-.05,800), yticks=[0,200,400,600,800]
    ) for ax in axes[1]]
axes_twin[-1].set(ylabel=r"$\alpha$", yticks=[0,1,2,3,4])
[ax.set(ylim=(4*-.05,4), yticks=[0,1,2,3,4], yticklabels=[]) for ax in axes_twin[:-1]]
[ax.text(-0,1.1, "{}".format(txt), transform=ax.transAxes) for ax,txt in zip(axes[0],["(a)","(b)","(c)"])]

fig.tight_layout()
fig.subplots_adjust(hspace=.1,wspace=.2)
# plt.show()
fig.savefig("fig_selfsustain.png", dpi=400)
fig.savefig("fig_selfsustain.pdf", dpi=400)