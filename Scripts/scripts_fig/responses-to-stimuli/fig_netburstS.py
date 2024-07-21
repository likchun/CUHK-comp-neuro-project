from preamble import *


fig, [ax1,ax2,ax3] = plt.subplots(1, 3, figsize=(12,4), sharex=True)
xticks = [0,20,40,60,80,100]
stimuli = Stimuli([20,40,60,80,100])

for f,sty,c in zip([
    "data_netburstS_e02i02.npy",
    "data_netburstS_e02i06.npy",
    "data_netburstS_e03i02.npy",
],[".-",".--",".-."],["g","darkorange","m"]):
    results = np.load(os.path.join(data_path.DA_resptostim, f), allow_pickle=True)

    f_nb = np.array(results[2], dtype=float)
    mean_INBI_ms = np.array(results[3], dtype=float)
    std_INBI_ms = np.array(results[4], dtype=float)
    mean_K_nb = np.array(results[5], dtype=float)
    std_K_nb = np.array(results[6], dtype=float)

    ax1.plot([0]+stimuli.S, f_nb, sty, c=c, ms=myMarkerSize0)
    ax1.set(xlabel="$S$", ylabel="$f_{nb}$ (Hz)")

    ax2.errorbar([0]+stimuli.S, mean_INBI_ms, std_INBI_ms,
                fmt=sty, c=c, ms=myMarkerSize0, capsize=4, label="interval")
    ax2.set(xlabel="$S$", ylabel="INBI (ms)")

    ax3.errorbar([0]+stimuli.S, mean_K_nb/500, std_K_nb/500,
                fmt=sty, c=c, ms=myMarkerSize0, capsize=4)
    ax3.set(xlabel="$S$", ylabel="$k_{nb}$")

[ax.set(xlim=(-5,105)) for ax in [ax1,ax2,ax3]]
ax1.set(ylim=(0,10))
ax2.set(ylim=(0,250))
ax3.set(ylim=(0,6000/500))

fig.tight_layout()
fig.subplots_adjust(wspace=.5)
# plt.show()
fig.savefig("fig_netburstS.pdf", dpi=400)
fig.savefig("fig_netburstS.png", dpi=400)