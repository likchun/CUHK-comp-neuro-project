from preamble import *
from scipy import optimize
import json


C_thr = 0.03

fig, ax = plt.subplots(figsize=[6,5])
xlim = (0, 0.8)

dumpdata = json.load(open(os.path.join(data_path.DA_phasediag, "data_phasediagA.json"), 'r', encoding="utf-8"))
gIs = [x["gI"] for x in dumpdata]
gEs_gI = [x["gEs"] for x in dumpdata]
Cs_gI = [x["Cs"] for x in dumpdata]

# Phase diagram for fixed adaptation level = 1
c_argmax = [int(np.argmax(c)) for c in Cs_gI]
gE_thrL = [np.interp(C_thr, np.array(c[:c_argmax[i]+1]), np.array(gE[:c_argmax[i]+1])) for i,(c,gE) in enumerate(zip(Cs_gI,gEs_gI))]
gE_thrR = [np.interp(C_thr, np.array(c[c_argmax[i]:])[::-1], np.array(gE[c_argmax[i]:])[::-1]) for i,(c,gE) in enumerate(zip(Cs_gI,gEs_gI))]
ax.plot(gE_thrL, gIs, "k.--")
ax.plot(gE_thrR, gIs, "k.--")
ax.text(.025,.3, "I", fontdict=dict(font="charter", size=myFontSize0), ha="center", va="center", transform=ax.transData, bbox=dict(fc="none", ec="none", pad=5))
ax.text(.26, .3, "II", fontdict=dict(font="charter", size=myFontSize0+10), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.text(.68, .3, "III", fontdict=dict(font="charter", size=myFontSize0+5), ha="center", va="center", transform=ax.transData, bbox=dict(fc="w", ec="none", pad=5))
ax.set(xlabel="$g_E$", ylabel="$g_I$", xlim=xlim, ylim=(0,.6), xticks=np.arange(0,xlim[-1]+.1,.1))

fig.tight_layout()
# plt.show()
fig.savefig("fig_phasediagA-1.pdf", dpi=400)
fig.savefig("fig_phasediagA-1.png", dpi=400)