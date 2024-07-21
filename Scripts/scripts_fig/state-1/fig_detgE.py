from preamble import *
from import_theofrS1 import theofrS1


fig, ax = plt.subplots(figsize=(7,4))
colors = ["darkorange", "g", "m", "royalblue"]
ls = ["-", "-.", "--", ":"]
ge = np.arange(0,.2001,0.0001)
for i,gi in enumerate([8,.6,.2,0]):
    det = [theofrS1.get_determinant(g,gi) for g in ge]
    ax.plot(ge, det, "-", c=colors[i], ls=ls[i], lw=2, label="{}".format(gi))
ax.set(xlabel="$g_E$", ylabel="$\Delta$")
ax.set(xlim=(0,.18), ylim=(-8,2))
ax.set(xticks=[0,.02,.04,.06,.08,.1,.12,.14,.16,.18])
ax.legend(["8.0","0.6","0.2","0.0"], title="$g_I$")
fig.tight_layout()
# plt.show()
fig.savefig("fig_detgE.pdf", dpi=300)
fig.savefig("fig_detgE.png", dpi=300)