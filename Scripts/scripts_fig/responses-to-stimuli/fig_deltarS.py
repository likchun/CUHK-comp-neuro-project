from preamble import *
stimuli = Stimuli([20,40,60,80,100])


frs = np.load(os.path.join(data_path.DA_resptostim, "data_deltarS.npy"))
frs_chg = np.array([fr - fr[0] for fr in frs])


fig, ax = plt.subplots(figsize=(6,5))
stys = ["rs-", "rs-", "mD-", "mD-", "b^-", "b^-", "go-", "go-"]
mfcs = ["r", "none", "m", "none", "b", "none", "g", "none"]
[ax.plot([0]+stimuli.S, fc, sty, mfc=mfc, ms=8, lw=1) for fc,sty,mfc in zip(frs_chg,stys,mfcs)]
ax.set(xlim=(-5,105), ylim=(-2,12))
ax.set(xlabel="$S$", ylabel="$\Delta r$ (Hz)")

fig.tight_layout()
# plt.show()
fig.savefig("fig_deltarS.pdf", dpi=400)
fig.savefig("fig_deltarS.png", dpi=400)