from preamble import *
from scipy import stats
from mylib import NeuronalNetwork


net = NeuronalNetwork()
net.adjacency_matrix_from_file(data_path.network_A)

fig,axes = plt.subplots(1,2,figsize=(8,4))
qgraph.bar_chart_INT(net.in_degree_exc, ax=axes[0], label="EXC", fc="w", ec="k", hatch="//")
qgraph.bar_chart_INT(net.in_degree_inh, ax=axes[0], label="INH", fc="w", ec="k", hatch="o.")
outdeg_x,outdeg_y = qgraph.bar_chart_INT(net.out_degree, color="k", ax=axes[1])
fit_x = np.linspace(0,25,100)
# Binomial distribution (see: https://en.wikipedia.org/wiki/Degree_distribution)
fit_y = stats.binom.pmf(fit_x, 1000-1, 0.01)
axes[1].plot(fit_x, fit_y*1000, "--", c="k", lw=2)
axes[0].set(xlabel="$k$",ylabel="number of occurrence")
axes[1].set(xlabel="$k'$")
axes[0].set(xticks=[0,2,4,6,8,10])
axes[1].set(xticks=[0,5,10,15,20,25], yticks=[0,25,50,75,100,125,150])
axes[0].set(xlim=(0,10),ylim=(0,1250))
axes[1].set(xlim=(0,25),ylim=(0,150))
axes[0].legend(fontsize=myFontSize3)

axes[0].text(.04,.86, "(a)", transform=axes[0].transAxes)
axes[1].text(.04,.86, "(b)", transform=axes[1].transAxes)

fig.tight_layout()
# plt.show()
fig.savefig("fig_degdistA.pdf", dpi=400)
fig.savefig("fig_degdistA.png", dpi=400)