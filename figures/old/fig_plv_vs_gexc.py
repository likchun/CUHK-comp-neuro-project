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



gplv_data = []
for i, (w_i,w_e) in enumerate(itertools.product(w_inh,w_exc)):
    directory = directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alpha)
    (gplv,plvmat,_) = plv.load_plv_data(os.path.join(directory,"plv_data"))
    gplv_data.append(gplv)
gplv_data = np.array(gplv_data).reshape(len(w_inh),len(w_exc))


fig, ax = plt.subplots(figsize=(6,6))

_w_inh = [0]+w_inh[1:]
[graphing.line_plot(w_exc,gplv_data[i],c=graphing.mycolors[i],label="{}".format(_w_inh[i]),ax=ax) for i in range(len(w_inh))]
graphing.line_plot([w_exc[0],w_exc[-1]],[plv_thresh,plv_thresh],style="--",c="brown",ax=ax)
ax.set(xlim=(w_exc[0],w_exc[-1]),ylim=(0,1))
ax.legend(fontsize=16)
ax.set(xlabel="$g_{{exc}}$",ylabel="PLV")

plt.tight_layout()
plt.show()
# fig.savefig("fig_plv_vs_gexc",dpi=200)