from preamble import *
from mylib import power_spectral_density_normalized


fig, axes = plt.subplots(1, 2, figsize=[8,4], sharey=True)

params = [
    (0.04, 0.2, 3, 1),
    (0.04, 8,   3, 1),
]
fmt = ["k-", "k-"]
for i, (gE,gI,alpha,lmbd) in enumerate(params):
    nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(gE,gI,alpha,lmbd)))
    freq,spow = power_spectral_density_normalized(nd.dynamics.average_firing_rate_time_histogram(5, time_scale="s")[1], 1000/5)
    axes[i].plot(freq, spow, fmt[i], lw=1)
[ax.set(xlim=(0,40), xticks=[0,10,20,30,40], ylim=(None, .5), yticks=[0,.1,.2,.3,.4,.5]) for ax in axes]
# ax.set(xlim=(0,20), xticks=[0,5,10,15,20])
[ax.set(xlabel="$f$ (Hz)") for ax in axes]
axes[0].set(ylabel="$S_{rr}$")

axes[0].text(.85,.85, "(a)", transform=axes[0].transAxes)
axes[1].text(.85,.85, "(b)", transform=axes[1].transAxes)
fig.tight_layout()
# plt.show()
fig.savefig("fig_specpowerS1.pdf", dpi=400)
fig.savefig("fig_specpowerS1.png", dpi=400)
