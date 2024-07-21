from preamble import *


fig, ax = plt.subplots(figsize=(6,5))
# fig, ax = plt.subplots(figsize=(4.5,3.5))

params = [
    (0.02, 0.2, 3, 1),
    (0.03, 0.2, 3, 1),
    (0.04, 0.2, 3, 1),
    (0.05, 0.2, 3, 1),
    (0.06, 0.2, 3, 1),
    (0.08, 0.2, 3, 1),
    (0.1,  0.2, 3, 1),
]
Cs = []
for i, (gE,gI,alpha,kappa) in enumerate(params):
    nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    nd.retain_dynamics(1000,2000)
    Cs.append(nd.dynamics.analysis.coherence_parameter(32))
ax.plot([p[0] for p in params], Cs, "ks--", label="$\kappa=1$")


params = [
    (0.02, 0.2, 3, 0),
    (0.03, 0.2, 3, 0),
    (0.04, 0.2, 3, 0),
    (0.05, 0.2, 3, 0),
    (0.06, 0.2, 3, 0),
    (0.08, 0.2, 3, 0),
    (0.1,  0.2, 3, 0),
]
Cs = []
for i, (gE,gI,alpha,kappa) in enumerate(params):
    nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(gE,gI,alpha,kappa)))
    nd.retain_dynamics(1000,2000)
    Cs.append(nd.dynamics.analysis.coherence_parameter(32))
ax.plot([p[0] for p in params], Cs, "r^-", label="$\kappa=0$")


ax.plot([.02,.1], [.03,.03], "k:")
ax.set(xlabel="$g_E$", ylabel="$C$")
ax.set(ylim=(1*-.03,1), xlim=(.02,.1))
ax.legend()
fig.tight_layout()
# fig.subplots_adjust(hspace=.05, wspace=.3)
# plt.show()
fig.savefig("fig_CgEkappa.pdf", dpi=400)
fig.savefig("fig_CgEkappa.png", dpi=400)