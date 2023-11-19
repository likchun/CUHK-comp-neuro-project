from mylib3 import *
from plvlib import *
plv = PhaseLockingValue()
graphing.modern_style()
graphing.default_legend_style()


w_inh = [1e-8,.2,.4,.6]
w_exc1 = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2]
# w_exc0 = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2,.3,.35,.4,.42,.45,.5,.6]
w_exc0 = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2,.3,.35,.4,.45,.5,.6]
alphas = [5]


directory0 = "/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"
# cmap = "Greens"
cmap = "Greys"


gplv_data0,gplv_data1 = [],[]
for i, (w_i,w_e,a) in enumerate(itertools.product(w_inh,w_exc0,alphas)):
    directory = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,a)
    (gplv,plvmat,_) = plv.load_plv_data(os.path.join(directory,"plv_data"))
    gplv_data0.append(gplv)
gplv_data0 = np.array(gplv_data0).reshape(len(w_inh),len(w_exc0))
for i, (w_i,w_e,a) in enumerate(itertools.product(w_inh,w_exc1,alphas)):
    directory = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,a)
    (gplv,plvmat,_) = plv.load_plv_data(os.path.join(directory,"plv_data"))
    gplv_data1.append(gplv)
gplv_data1 = np.array(gplv_data1).reshape(len(w_inh),len(w_exc1))

fig, axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
we0,wi = np.meshgrid(w_exc0,w_inh)
pcm = axes[0].pcolormesh(we0,wi,gplv_data0,cmap=cmap,vmin=0,vmax=1,snap=True)
we1,wi = np.meshgrid(w_exc1,w_inh)
pcm = axes[1].pcolormesh(we1,wi,gplv_data1,cmap=cmap,vmin=0,vmax=1,snap=True)
# for i, (w_i,w_e,a) in enumerate(itertools.product(w_inh,w_exc,alphas)):
#     ax.text(w_e,w_i,"{}".format(str(gplv_data.flat[i])[1:4]),fontdict=dict(size=8),horizontalalignment="center",verticalalignment="center")
#     if w_e > 0.18: ax.text(w_e,w_i,"{}".format(str(gplv_data.flat[i])[1:4]),fontdict=dict(size=8),horizontalalignment="center",verticalalignment="center")
fig.colorbar(pcm,label="global PLV")
axes[1].set(xlim=(w_exc1[0]-.01,w_exc1[-1]+.01),ylim=(w_inh[0]-.05,w_inh[-1]+.05))
axes[0].set(xlim=(w_exc0[0]-.01,w_exc0[-1]+.05),ylim=(w_inh[0]-.05,w_inh[-1]+.05))
axes[0].set(xlabel="$g_{{exc}}$",ylabel="$g_{{inh}}$")
axes[1].set(xlabel="$g_{{exc}}$")
axes[0].text(.04,.45,"I",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
axes[0].text(.25,.3,"II",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
axes[0].text(.55,.15,"III",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
axes[1].text(.04,.5,"I",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
axes[1].text(.15,.1,"II",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
axes[1].set_xticks([0,.06,.12,.18])
axes[0].set_xticks([0,.1,.2,.3,.4,.5,.6])
axes[0].set_yticks([0,.2,.4,.6])
axes[1].set_title("blown-up for small $g_{{exc}}$",fontdict=dict(size=20),loc="left",pad=10)

plt.tight_layout()
plt.show()
# fig.savefig("fig_phase_diagram_plv",dpi=200)
# fig.savefig("fig_phase_diagram_plv",bbox_inches="tight",pad_inches=.5,dpi=200)