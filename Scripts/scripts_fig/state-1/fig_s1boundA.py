from preamble import *
from scipy import optimize
import json


C_thr = 0.03

fig, ax = plt.subplots(figsize=[6,4.5])
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

# Boundary obtained from calculations
pEs = np.load(os.path.join(data_path.DA_state1_pXY, "gE,pEE,pIE.npy"))
pIs = np.load(os.path.join(data_path.DA_state1_pXY, "gI,pEI,pII.npy"))
gEs, ps_EE, ps_IE = pEs[0], pEs[1], pEs[2]
gIs, ps_EI, ps_II = pIs[0], pIs[1], pIs[2]
k_E, k_I = 8, 2

def find_determinant(gE):
    p_EE = np.interp(gE, gEs, ps_EE)
    p_IE = np.interp(gE, gEs, ps_IE)
    return k_E*k_I*p_EE*p_II - k_E*k_I*p_IE*p_EI - k_E*p_EE - k_I*p_II + 1

gI_all = np.concatenate([np.arange(0,0.2,0.02),np.arange(0.2,3,0.02)])
gE_roots = []
for gI in gI_all:
    p_EI = np.interp(gI, gIs, ps_EI)
    p_II = np.interp(gI, gIs, ps_II)
    result = optimize.root_scalar(find_determinant, x0=0.01, x1=0.1)
    gE_roots.append(result.root)
# Picture predict upper bound
ax.plot(gE_roots, gI_all, "r.-", ms=3)

fig.tight_layout()
# plt.show()
fig.savefig("fig_s1boundA.pdf", dpi=400)
fig.savefig("fig_s1boundA.png", dpi=400)