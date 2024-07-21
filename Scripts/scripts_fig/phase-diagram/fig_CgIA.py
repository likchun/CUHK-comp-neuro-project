from preamble import *


C_thr = 0.03
binsize_ms = 32

gEs = [.6, .3, .2, .1, .04]
gIs_gE_str = []


for gE in gEs:
    params = [tuple(map(str, name.split(","))) for name in os.listdir(data_path.SD_netA_mapout_gEgI) if os.path.isdir(os.path.join(data_path.spont_activ_netA, name))]
    params = list(filter(lambda par: par[0]=="{}".format(gE), params))
    params.sort(key=lambda par: float(par[1]))
    gIs_gE_str.append([par[1] for par in params])
params = [[(gE, gI, 3, 1) for gI in gIs_gE_str[i]] for i,gE in enumerate(gEs)]

nds = [[NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*gEIal))) for gEIal in par] for par in params]
[[_nd.remove_dynamics(500,0) for _nd in nd] for nd in nds]
Cs_gE = [[_nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms) for _nd in nd] for nd in nds]
gIs_gE = [np.array([float(x[1]) for x in par]) for par in params]

fig, ax = plt.subplots(figsize=[6,5])
# fig, ax = plt.subplots(figsize=[5,3.5])
fmt = ["ro-", "ms--", "bd--", "cp--", "k^-."]
xlim = (0,8)
ax.plot(xlim, [C_thr,C_thr], "k:", lw=1.5)
[ax.plot(gIs_gE[i], Cs_gE[i], fmt[k], ms=8, lw=1.5, label="{}".format(gEs[i])) for k,i in enumerate(range(len(gEs)))]
ax.set(xlabel="$g_I$", ylabel="$C$", xlim=xlim, xticks=np.linspace(0,8,9), ylim=(None, 1))
ax.legend(title="$g_E$", loc="center left", bbox_to_anchor=(1,0.5))

fig.tight_layout()
# plt.show()
fig.savefig("fig_CgI.pdf", dpi=400)
fig.savefig("fig_CgI.png", dpi=400)