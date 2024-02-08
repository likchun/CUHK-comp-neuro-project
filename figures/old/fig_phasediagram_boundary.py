from mylib3 import *
from plvlib import *
plv = PhaseLockingValue()
graphing.modern_style()
graphing.default_legend_style()


w_inh = [1e-8,.2,.4,.6]
w_exc = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2,.3,.35,.4,.42,.45,.5,.6]
alpha = 5
plv_thresh = .25


directory0 = "/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"


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

fig, ax = plt.subplots(figsize=(6,6))
graphing.line_plot(crit_wexcL,w_inh,style=".-",c="k",ax=ax)
graphing.line_plot(crit_wexcR,w_inh,style=".-",c="k",ax=ax)

ax.set(xlim=(0,.7),ylim=(0,.6))
ax.set(xticks=[0,.2,.4,.6],yticks=[0,.2,.4,.6])
ax.set(xlabel="$g_{{exc}}$",ylabel="$g_{{inh}}$")
ax.text(.06,.5,"I",fontdict=dict(font="charter",fontsize=30),ha="center",va="center")
ax.text(.27,.3,"II",fontdict=dict(font="charter",fontsize=30),ha="center",va="center")
ax.text(.58,.15,"III",fontdict=dict(font="charter",fontsize=30),ha="center",va="center")

plt.tight_layout()
plt.show()
# fig.savefig("fig_phase_diagram_boundary",dpi=200)
# fig.savefig("fig_phase_diagram_boundary",bbox_inches="tight",pad_inches=.5,dpi=150)
