from preamble import *


stimuli = Stimuli([20, 40, 60, 80, 100])

Cs = np.load(os.path.join(data_path.DA_resptostim, "data_CS.npy"))

fig, ax = plt.subplots(figsize=(6,5))
stys = ["rs-", "rs--", "mD-", "mD--", "b^-", "b^--", "go-", "go--"]
[ax.plot([0]+stimuli.S, C, sty, ms=6) for C,sty in zip(Cs,stys)]
ax.plot([0,stimuli.S[-1]], [.03,.03], "k:")

ax.set(xlim=(0,100))
ax.set(ylim=(-.05,1))
ax.set(xlabel="$S$", ylabel="$C$")

fig.tight_layout()
fig.subplots_adjust(wspace=.2)
# plt.show()
fig.savefig("fig_CS.pdf", dpi=400)
fig.savefig("fig_CS.png", dpi=400)