from preamble import *
import json


C_thr = 0.03

dumpdata = json.load(open(os.path.join(data_path.DA_phasediag, "data_phasediagA.json"), 'r', encoding="utf-8"))
gIs = [x["gI"] for x in dumpdata]
gEs_gI = [x["gEs"] for x in dumpdata]
Cs_gI = [x["Cs"] for x in dumpdata]

# c_argmax = [int(np.argmax(c)) for c in Cs_gI]
# gE_thrL = [np.interp(C_thr, np.array(c[:c_argmax[i]+1]), np.array(gE[:c_argmax[i]+1])) for i,(c,gE) in enumerate(zip(Cs_gI,gEs_gI))]
# gE_thrR = [np.interp(C_thr, np.array(c[c_argmax[i]:])[::-1], np.array(gE[c_argmax[i]:])[::-1]) for i,(c,gE) in enumerate(zip(Cs_gI,gEs_gI))]

# Coherence oreder parameter C vs excitatory synaptic strength gE for fixed adaptation level = 1
fig, ax = plt.subplots(figsize=[8,4])
colors = ["darkorange", "g", "c", "m", "royalblue"]
fmt = ["o-", "s-", "D-", "^-", "p-"]
[ax.plot(gEs_gI[i], Cs_gI[i], fmt[k], c=colors[k], ms=6, lw=1.5, label="{}".format(gIs[i]), zorder=5) for k,i in enumerate([7,6,4,2,0])]
xlim = (0, 1)
ax.plot(xlim,[C_thr,C_thr], "k--", lw=1.5, zorder=5)
ax.set_xlabel("$g_E$")
ax.set_ylabel("$C$")
ax.set(xlim=xlim, xticks=np.arange(0,xlim[-1]+.1,.2), yticks=[0,.2,.4,.6,.8,1])
ax.legend(["8.0","0.6","0.4","0.2","0.0"], title="$g_I$")
ax.text(.025,.3, "I", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="none", ec="none", pad=5))
ax.text(.2,  .3, "II", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.text(.65, .3, "III", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
# gE_thr = [np.interp(C_thr, c, gE) for c,gE in zip(Cs_gI,gEs_gI)]
# ax.plot(gE_thr, np.full(len(gE_thr), C_thr), "ko", mec="none")

fig.tight_layout()
# plt.show()
fig.savefig("fig_CgE.pdf", dpi=400)
fig.savefig("fig_CgE.png", dpi=400)
