from preamble import *
from import_theofrS1 import theofrS1


fig, ax = plt.subplots(figsize=(6,4.5))
colors = ["k", "k"]
symbol = ["x", "+"]

gIs = [0.6, 0.2]
[ax.plot([], [], "-"+symbol[i], lw=1.8, ms=16, mew=1.4, c=colors[i]) for i in range(len(gIs))]
ax.legend(gIs, title="$g_I$")

ge = np.arange(0,.06001,0.0001)
for i,gi in enumerate(gIs):
    nus = [theofrS1.get_predicted_spike_rate(gE=x, gI=gi) for x in ge]
    ax.plot(ge, [x[0] for x in nus], "-",  c=colors[i], lw=1.8, label="{}".format(gi))
    ax.plot(ge, [x[1] for x in nus], "--", c=colors[i], lw=1.8, label="{}".format(gi))

    for _ge in [0,.01,.02,.03,.04,.05,.055]:
        try:
            nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},3,1".format(_ge,gi)))
            nd.remove_dynamics(200,0)
            nd.apply_neuron_mask(range(200,1000))
            ax.plot(_ge, nd.dynamics.mean_firing_rate.mean(), symbol[i], mfc="none", ms=16, mew=1.4, c=colors[i])
            nd.apply_neuron_mask(range(200))
            ax.plot(_ge, nd.dynamics.mean_firing_rate.mean(), symbol[i], mfc="none", ms=16, mew=1.4, c=colors[i])
        except FileNotFoundError: pass

ax.set(yscale="log")
ax.set(xlim=(-.001,0.06), ylim=(.1,100), xticks=[0,.01,.02,.03,.04,.05,.06])
ax.set(xlabel="$g_E$", ylabel="$r_{E(I)}$ (Hz)")
fig.tight_layout()
# plt.show()
fig.savefig("fig_frS1.pdf", dpi=400)
fig.savefig("fig_frS1.png", dpi=400)