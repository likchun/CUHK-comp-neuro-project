from preamble import *
import json


C_thr = 0.03

fig, ax = plt.subplots(figsize=[8,6])
xlim = (0, 0.8)

dumpdata = json.load(open(os.path.join(data_path.DA_phasediag, "data_phasediagB.json"), 'r', encoding="utf-8"))
gIs = [x["gI"] for x in dumpdata]
gEs_gI = [x["gEs"] for x in dumpdata]
Cs_gI = [x["Cs"] for x in dumpdata]

# Phase diagram for fixed adaptation level = 1
c_argmax = [int(np.argmax(c)) for c in Cs_gI]
gE_thrL = [np.interp(C_thr, np.array(c[:c_argmax[i]+1]), np.array(gE[:c_argmax[i]+1])) for i,(c,gE) in enumerate(zip(Cs_gI,gEs_gI))]
gE_thrR = [np.interp(C_thr, np.array(c[c_argmax[i]:])[::-1], np.array(gE[c_argmax[i]:])[::-1]) for i,(c,gE) in enumerate(zip(Cs_gI,gEs_gI))]
ax.plot(gE_thrL, gIs, "k.--")
ax.plot(gE_thrR, gIs, "k.--")
ax.text(.025,.2, "I", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="none", ec="none", pad=5))
ax.text(.26, .2, "II", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.text(.62, .2, "III", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.set(xlabel="$g_E$", ylabel="$g_I$", xlim=xlim, ylim=(0,.6), xticks=np.arange(0,xlim[-1]+.1,.1))

# Inset: Coherence oreder parameter C vs excitatory synaptic strength gE for fixed adaptation level = 1
gE_thr = [np.interp(C_thr, c, gE) for c,gE in zip(Cs_gI,gEs_gI)]
ax_inset = ax.inset_axes([0.21, 0.6, 0.4, 0.35])
colors = ["g", "m"]
fmt = ["s-", "^-"]
[ax_inset.plot(gEs_gI[i], Cs_gI[i], fmt[k], c=colors[k], ms=4, lw=1, label="{}".format(gIs[i])) for k,i in enumerate([3,1])]
ax_inset.tick_params(axis="both", which="major", labelsize=myFontSize3)
ax_inset.set_xlabel("$g_E$", fontsize=myFontSize2)
ax_inset.set_ylabel("$C$", fontsize=myFontSize2)
ax_inset.set(xlim=xlim, xticks=np.arange(0,xlim[-1]+.1,.2), yticks=[0,.2,.4,.6,.8,1])
ax_inset.legend(title="$g_I$", fontsize=myFontSize3, title_fontsize=myFontSize2)

fig.tight_layout()
# plt.show()
fig.savefig("fig_phasediagB.pdf", dpi=400)
fig.savefig("fig_phasediagB.png", dpi=400)