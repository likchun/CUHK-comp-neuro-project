from mylib3 import *
from plvlib import *
plv = PhaseLockingValue()
graphing.modern_style()
graphing.default_legend_style()


w_inh = [1e-8,.2,.4,.6]
w_exc = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2,.3,.35,.4,.45,.5,.6]
alphas = [3,5,7]


from mylib3 import *
from plvlib import *
plv = PhaseLockingValue()
graphing.modern_style()
graphing.default_legend_style()


w_inh = [1e-8,.2,.4,.6]
w_exc = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2,.3,.35,.4,.45,.5,.6]
plv_thresh = .25


directory0 = "/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"

fig, ax = plt.subplots(figsize=(6,6))

alpha = 5
crit_wexcL,crit_wexcR = [],[]
for w_i in w_inh:
    _prev = []
    for w_e in w_exc:
        filepath = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alpha)
        gplv = np.load(open(os.path.join(filepath,"plv_data"),"rb"),allow_pickle=True)[0]
        try:
            if gplv > plv_thresh and _prev[-1][1] < plv_thresh:
                crit_wexcL.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
        except IndexError: pass
        try:
            if gplv < plv_thresh and _prev[-1][1] > plv_thresh: 
                crit_wexcR.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
        except IndexError: pass
        _prev.append((w_e,gplv))
graphing.line_plot(crit_wexcL,w_inh,style=".-",c="k",ax=ax)
graphing.line_plot(crit_wexcR,w_inh,style=".-",c="k",ax=ax)

alpha = 3
crit_wexcL,crit_wexcR = [],[]
for w_i in w_inh:
    _prev = []
    for w_e in w_exc:
        filepath = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alpha)
        gplv = np.load(open(os.path.join(filepath,"plv_data"),"rb"),allow_pickle=True)[0]
        try:
            if gplv > plv_thresh and _prev[-1][1] < plv_thresh:
                crit_wexcL.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
        except IndexError: pass
        try:
            if gplv < plv_thresh and _prev[-1][1] > plv_thresh: 
                crit_wexcR.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
        except IndexError: pass
        _prev.append((w_e,gplv))
graphing.line_plot(crit_wexcL,w_inh,style=".-",c="g",ax=ax)
graphing.line_plot(crit_wexcR,w_inh,style=".-",c="g",ax=ax)

alpha = 7
crit_wexcL,crit_wexcR = [],[]
for w_i in w_inh:
    _prev = []
    for w_e in w_exc:
        filepath = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alpha)
        gplv = np.load(open(os.path.join(filepath,"plv_data"),"rb"),allow_pickle=True)[0]
        try:
            if gplv > plv_thresh and _prev[-1][1] < plv_thresh:
                crit_wexcL.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
        except IndexError: pass
        try:
            if gplv < plv_thresh and _prev[-1][1] > plv_thresh: 
                crit_wexcR.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
        except IndexError: pass
        _prev.append((w_e,gplv))
graphing.line_plot(crit_wexcL,w_inh,style=".-",c="r",ax=ax)
graphing.line_plot(crit_wexcR,w_inh,style=".-",c="r",ax=ax)


ax.set(xlim=(0,.7),ylim=(0,.6))
ax.set(xticks=[0,.2,.4,.6],yticks=[0,.2,.4,.6])
ax.set(xlabel="$g_{{exc}}$",ylabel="$g_{{inh}}$")
ax.text(.06,.5,"I",fontdict=dict(font="charter",fontsize=30),ha="center",va="center")
ax.text(.27,.3,"II",fontdict=dict(font="charter",fontsize=30),ha="center",va="center")
ax.text(.58,.15,"III",fontdict=dict(font="charter",fontsize=30),ha="center",va="center")

plt.tight_layout()
# plt.show()
fig.savefig("fig_phase_diagram_boundary_noise",dpi=200)
# fig.savefig("fig_phase_diagram_boundary",bbox_inches="tight",pad_inches=.5,dpi=150)


exit()



directory0 = "/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"
# cmap = "Greens"
cmap = "RdBu"


gplv_data_a3 = []
for i, (w_i,w_e) in enumerate(itertools.product(w_inh,w_exc)):
    directory = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alphas[0])
    (gplv,plvmat,_) = plv.load_plv_data(os.path.join(directory,"plv_data"))
    gplv_data_a3.append(gplv)
gplv_data_a3 = np.array(gplv_data_a3).reshape(len(w_inh),len(w_exc))

gplv_data_a5 = []
for i, (w_i,w_e) in enumerate(itertools.product(w_inh,w_exc)):
    directory = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alphas[1])
    (gplv,plvmat,_) = plv.load_plv_data(os.path.join(directory,"plv_data"))
    gplv_data_a5.append(gplv)
gplv_data_a5 = np.array(gplv_data_a5).reshape(len(w_inh),len(w_exc))

gplv_data_a7 = []
for i, (w_i,w_e) in enumerate(itertools.product(w_inh,w_exc)):
    directory = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alphas[2])
    (gplv,plvmat,_) = plv.load_plv_data(os.path.join(directory,"plv_data"))
    gplv_data_a7.append(gplv)
gplv_data_a7 = np.array(gplv_data_a7).reshape(len(w_inh),len(w_exc))


fig, axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
we0,wi = np.meshgrid(w_exc,w_inh)
pcm = axes[0].pcolormesh(we0,wi,gplv_data_a3-gplv_data_a5,cmap=cmap,vmin=-1,vmax=1,snap=True)
we1,wi = np.meshgrid(w_exc,w_inh)
pcm = axes[1].pcolormesh(we1,wi,gplv_data_a7-gplv_data_a5,cmap=cmap,vmin=-1,vmax=1,snap=True)
# for i, (w_i,w_e,a) in enumerate(itertools.product(w_inh,w_exc,alphas)):
#     ax.text(w_e,w_i,"{}".format(str(gplv_data.flat[i])[1:4]),fontdict=dict(size=8),horizontalalignment="center",verticalalignment="center")
#     if w_e > 0.18: ax.text(w_e,w_i,"{}".format(str(gplv_data.flat[i])[1:4]),fontdict=dict(size=8),horizontalalignment="center",verticalalignment="center")
# fig.colorbar(pcm,label="global PLV")
# axes[1].set(xlim=(w_exc1[0]-.01,w_exc1[-1]+.01),ylim=(w_inh[0]-.05,w_inh[-1]+.05))
# axes[0].set(xlim=(w_exc0[0]-.01,w_exc0[-1]+.05),ylim=(w_inh[0]-.05,w_inh[-1]+.05))
axes[0].set(xlabel="$g_{{exc}}$",ylabel="$g_{{inh}}$")
axes[1].set(xlabel="$g_{{exc}}$")
# axes[0].text(.04,.45,"I",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
# axes[0].text(.25,.3,"II",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
# axes[0].text(.55,.15,"III",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
# axes[1].text(.04,.5,"I",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
# axes[1].text(.15,.1,"II",ha="center",va="center",font="charter",bbox=dict(boxstyle="circle",fc="w",ec="k",alpha=.9))
# axes[1].set_xticks([0,.06,.12,.18])
# axes[0].set_xticks([0,.1,.2,.3,.4,.5,.6])
# axes[0].set_yticks([0,.2,.4,.6])
# axes[1].set_title("blown-up for small $g_{{exc}}$",fontdict=dict(size=20),loc="left",pad=10)

plt.tight_layout()
plt.show()
# fig.savefig("fig_phase_diagram_plv",dpi=200)
# fig.savefig("fig_phase_diagram_plv",bbox_inches="tight",pad_inches=.5,dpi=200)