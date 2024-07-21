from preamble import *


fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(8,4))

params = [
    (0.2, 0,   3, 1),
    (0.2, 0.1, 3, 1),
    (0.2, 0.2, 3, 1),
    (0.2, 0.3, 3, 1),
    (0.2, 0.4, 3, 1),
    (0.2, 0.6, 3, 1),
    (0.2, 0.8, 3, 1),
    (0.2, 1,   3, 1),
    (0.2, 1.5, 3, 1),
    (0.2, 2,   3, 1),
    (0.2, 2.5, 3, 1),
    (0.2, 3,   3, 1),
    (0.2, 3.5, 3, 1),
    (0.2, 4,   3, 1),
]
data = np.load(os.path.join(data_path.DA_resptostim, "data_entropygI,state2.npy"), allow_pickle=True)
MIs = data[:,0]
Hs  = data[:,1]

EIratio = [par[1]/par[0] for par in params]
gIs = [par[1] for par in params]

ax1.plot(gIs, Hs, "ko", label="0.2")
ax2.plot(gIs, MIs, "ko", label="0.2")

ax1.set_xscale("log")
ax2.set_xscale("log")
ax1.set_xticks([.1,.3,1,4])
ax2.set_xticks([.1,.3,1,4])
ax1.set(xlim=(None,4), ylim=(8*-.04,8))
ax2.set(xlim=(None,4), ylim=(.8*-.04,.8))
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax1.set(xlabel="$g_I$", ylabel="$H$ (bits)")
ax2.set(xlabel="$g_I$", ylabel="$I_m$ (bits)")
ax1.text(.05, .875, "(a)", transform=ax1.transAxes)
ax2.text(.05, .875, "(b)", transform=ax2.transAxes)

# ax1.legend(title="$g_E$", loc="lower right")
fig.tight_layout()
# plt.show()
fig.savefig("fig_entropygI.pdf", dpi=400)
fig.savefig("fig_entropygI.png", dpi=400)