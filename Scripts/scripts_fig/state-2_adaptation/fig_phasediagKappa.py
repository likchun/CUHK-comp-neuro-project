from preamble import *


C_thr = 0.0305
binsize_ms = 32


fig, ax = plt.subplots(figsize=[7.5,5])
ax.plot([],[],"k.:",label="0.6")
ax.plot([],[],"k.--",label="0.2")
ax.plot([],[],"k.-.",label="0.0")
ax.legend(title="$g_I$", loc="upper left", bbox_to_anchor=(1,1.025))


params = [[
        (0.06, 0.6, 3, 0),
        (0.06, 0.6, 3, 0.05),
        (0.06, 0.6, 3, 0.1),
        (0.06, 0.6, 3, 0.3),
        (0.06, 0.6, 3, 0.5),
    ],[
        (0.2,  0.6, 3, 0.2),
        (0.2,  0.6, 3, 0.3),
        (0.2,  0.6, 3, 0.4),
        (0.2,  0.6, 3, 0.5),
        (0.2,  0.6, 3, 1),
    ],[
        (0.4,  0.6, 3, 0.4),
        (0.4,  0.6, 3, 0.8),
        (0.4,  0.6, 3, 1),
        (0.4,  0.6, 3, 1.2),
    ],[
        (0.6,  0.6, 3, 0.6),
        (0.6,  0.6, 3, 1),
        (0.6,  0.6, 3, 1.3),
        (0.6,  0.6, 3, 2),
        (0.6,  0.6, 3, 4),
]]
Cs = []
for param in params:
    _Cs = []
    for par in param:
        try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(*par)))
        except FileNotFoundError: nd = NeuroData(os.path.join(data_path.spont_activ_netA, "{},{},{},{}".format(*par)))
        nd.remove_dynamics(500,0)
        _Cs.append(nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms))
    Cs.append(_Cs)
kappas = [[par[3] for par in param] for param in params]
kappa_thr = [np.interp(C_thr, Cs[i], kappas[i]) for i in range(len(Cs))]
ax.plot([param[0][0] for param in params], kappa_thr, "k.:", label="{}".format(0.6), lw=1.5)


params = [[
        (0.04, 0.6, 3, 0.1),
        (0.05, 0.6, 3, 0.1),
        (0.06, 0.6, 3, 0.1),
    ],[
        (0.04, 0.6, 3, 0.5),
        (0.05, 0.6, 3, 0.5),
        (0.06, 0.6, 3, 0.5),
    ],[
        (0.04, 0.6, 3, 1),
        (0.05, 0.6, 3, 1),
        # (0.06, 0.6, 3, 1),
    ],[
        (0.04, 0.6, 3, 2),
        (0.05, 0.6, 3, 2),
        (0.06, 0.6, 3, 2),
    ],[
        (0.04, 0.6, 3, 3),
        (0.05, 0.6, 3, 3),
        (0.06, 0.6, 3, 3),
]]
Cs = []
for param in params:
    _Cs = []
    for par in param:
        try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(*par)))
        except FileNotFoundError: nd = NeuroData(os.path.join(data_path.spont_activ_netA, "{},{},{},{}".format(*par)))
        nd.remove_dynamics(500,0)
        _Cs.append(nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms))
    Cs.append(_Cs)
gEs = [[par[0] for par in param] for param in params]
gE_thr = [np.interp(C_thr, Cs[i], gEs[i]) for i in range(len(params))]
ax.plot(gE_thr, [param[0][3] for param in params], "k.:", lw=1.5)


params = [[
        (0.055, 0.2, 3, 0.01),
        (0.055, 0.2, 3, 0.02),
    ],[
        (0.06, 0.2, 3, 0),
        (0.06, 0.2, 3, 0.02),
        (0.06, 0.2, 3, 0.05),
        (0.06, 0.2, 3, 0.1),
        (0.06, 0.2, 3, 0.3),
        (0.06, 0.2, 3, 0.5),
        (0.06, 0.2, 3, 1),
    ],[
        (0.2,  0.2, 3, 0.2),
        (0.2,  0.2, 3, 0.3),
        (0.2,  0.2, 3, 0.4),
        (0.2,  0.2, 3, 1),
    ],[
        (0.4,  0.2, 3, 0.6),
        (0.4,  0.2, 3, 0.8),
        (0.4,  0.2, 3, 1),
        (0.4,  0.2, 3, 1.2),
    ],[
        (0.6,  0.2, 3, 1),
        (0.6,  0.2, 3, 1.3),
        (0.6,  0.2, 3, 1.6),
        (0.6,  0.2, 3, 1.8),
]]
Cs = []
for param in params:
    _Cs = []
    for par in param:
        try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(*par)))
        except FileNotFoundError: nd = NeuroData(os.path.join(data_path.spont_activ_netA, "{},{},{},{}".format(*par)))
        nd.remove_dynamics(500,0)
        _Cs.append(nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms))
    Cs.append(_Cs)
kappas = [[par[3] for par in param] for param in params]
kappa_thr = [np.interp(C_thr, Cs[i], kappas[i]) for i in range(len(Cs))]
ax.plot([param[0][0] for param in params], kappa_thr, "k.--", label="{}".format(0.2), lw=1.5)


params = [[
        # (0.04, 0.2, 3, 0.1),
        (0.05, 0.2, 3, 0.1),
        (0.06, 0.2, 3, 0.1),
    ],[
        (0.04, 0.2, 3, 0.5),
        (0.05, 0.2, 3, 0.5),
        (0.06, 0.2, 3, 0.5),
    ],[
        (0.04, 0.2, 3, 1),
        (0.05, 0.2, 3, 1),
        (0.06, 0.2, 3, 1),
    ],[
        (0.04, 0.2, 3, 2),
        (0.05, 0.2, 3, 2),
        (0.06, 0.2, 3, 2),
    ],[
        (0.04, 0.2, 3, 3),
        (0.05, 0.2, 3, 3),
        (0.06, 0.2, 3, 3),
]]
Cs = []
for param in params:
    _Cs = []
    for par in param:
        try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(*par)))
        except FileNotFoundError: nd = NeuroData(os.path.join(data_path.spont_activ_netA, "{},{},{},{}".format(*par)))
        nd.remove_dynamics(500,0)
        _Cs.append(nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms))
    Cs.append(_Cs)
gEs = [[par[0] for par in param] for param in params]
gE_thr = [np.interp(C_thr, Cs[i], gEs[i]) for i in range(len(params))]
ax.plot([0.05]+gE_thr, [0.02]+[param[0][3] for param in params], "k.--", lw=1.5)


params = [[
        (0.055, 0, 3, 0.05),
        (0.055, 0, 3, 0.1),
        (0.055, 0, 3, 0.15),
    ],[
        (0.06, 0,  3, 0),
        (0.06, 0,  3, 0.1),
        (0.06, 0,  3, 0.3),
        (0.06, 0,  3, 0.5),
    ],[
        (0.2,  0,  3, 0.5),
        (0.2,  0,  3, 0.6),
        (0.2,  0,  3, 1),
    ],[
        (0.4,  0,  3, 1),
        (0.4,  0,  3, 1.1),
        (0.4,  0,  3, 1.2),
    ],[
        (0.6,  0,  3, 1),
        (0.6,  0,  3, 1.8),
        (0.6,  0,  3, 1.9),
]]
Cs = []
for param in params:
    _Cs = []
    for par in param:
        try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(*par)))
        except FileNotFoundError: nd = NeuroData(os.path.join(data_path.spont_activ_netA, "{},{},{},{}".format(*par)))
        nd.remove_dynamics(500,0)
        _Cs.append(nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms))
    Cs.append(_Cs)
kappas = [[par[3] for par in param] for param in params]
kappa_thr = [np.interp(C_thr, Cs[i], kappas[i]) for i in range(len(Cs))]
ax.plot([param[0][0] for param in params], kappa_thr, "k.-.", label="{}".format(0), lw=1.5)


params = [[
    #     (0.04, 0,   3, 0),
    #     (0.05, 0,   3, 0),
    #     (0.06, 0,   3, 0),
    # ],[
    #     (0.04, 0,   3, 0.1),
    #     (0.05, 0,   3, 0.1),
    #     (0.06, 0,   3, 0.1),
    # ],[
        (0.04, 0,   3, 0.5),
        (0.05, 0,   3, 0.5),
        (0.06, 0,   3, 0.5),
    ],[
        (0.04, 0,   3, 1),
        (0.05, 0,   3, 1),
        # (0.06, 0,   3, 1),
    ],[
        (0.04, 0,   3, 2),
        (0.05, 0,   3, 2),
        (0.06, 0,   3, 2),
    ],[
        (0.04, 0,   3, 3),
        (0.05, 0,   3, 3),
        (0.06, 0,   3, 3),
]]
Cs = []
for param in params:
    _Cs = []
    for par in param:
        try: nd = NeuroData(os.path.join(data_path.SD_netA_mapout_adap, "{},{},{},{}".format(*par)))
        except FileNotFoundError: nd = NeuroData(os.path.join(data_path.spont_activ_netA, "{},{},{},{}".format(*par)))
        nd.remove_dynamics(500,0)
        _Cs.append(nd.dynamics.analysis.coherence_parameter(binsize_ms=binsize_ms))
    Cs.append(_Cs)
gEs = [[par[0] for par in param] for param in params]
gE_thr = [np.interp(C_thr, Cs[i], gEs[i]) for i in range(len(params))]
ax.plot(gE_thr, [param[0][3] for param in params], "k.-.", lw=1.5)


ax.text(.025, 1.5, "I", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.text(.25, 1.5, "II", fontdict=dict(font="charter", size=myFontSize0+8), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.text(.52, .35, "III", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.set(xlabel="$g_E$", ylabel="$\\kappa$", xlim=(0,.6), ylim=(0,3))

fig.tight_layout()
# plt.show()
fig.savefig("fig_phasediagKappa.pdf", dpi=400)
fig.savefig("fig_phasediagKappa.png", dpi=400)