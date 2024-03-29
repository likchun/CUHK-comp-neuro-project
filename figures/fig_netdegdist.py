import os, sys
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir,".."))
from scripts.libs.mylib import qgraph, NeuronalNetwork
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
qgraph.config_font("avenir",24)
qgraph.default_legend_style()


net = NeuronalNetwork()
net.adjacency_matrix_from_file("networks/net_unifindeg.txt")

fig,axes = plt.subplots(1,2,figsize=(12,6))
qgraph.bar_chart_INT(net.in_degree_exc, ax=axes[0], label="EXC", fc="w", ec="k", hatch="//")
qgraph.bar_chart_INT(net.in_degree_inh, ax=axes[0], label="INH", fc="w", ec="k", hatch="o.")
outdeg_x,outdeg_y = qgraph.bar_chart_INT(net.out_degree, color="k", ax=axes[1])
m_fit,s_fit = stats.norm.fit(net.out_degree)
print(m_fit,s_fit)
gausfit_x = np.linspace(0,25,100)
gausfit_y = stats.norm.pdf(gausfit_x,m_fit,s_fit)
axes[1].plot(gausfit_x, gausfit_y*1000, "--", c="k", lw=2)
axes[0].set(xlabel="in-degree $k_{{in}}$",ylabel="number of occurrence")
axes[1].set(xlabel="out-degree $k_{{out}}$")
axes[0].set(xticks=[0,2,4,6,8,10])
axes[0].set(xlim=(0,10),ylim=(0,None))
axes[1].set(xlim=(0,23),ylim=(0,None))
axes[0].legend()

fig.tight_layout()
# plt.show()
fig.savefig("figures/fig_net_unifindeg_degdist.pdf", dpi=300)
fig.savefig("figures/fig_net_unifindeg_degdist.png", dpi=300)




# net = NeuronalNetwork()
# net.adjacency_matrix_from_file("/Users/likchun/NeuroProject/networks/matrixSF.txt")

# fig,axes = plt.subplots(1,2,figsize=(12,6))
# indeg_x,indeg_y = graphing.bar_chart_INT(net.in_degree,c="k",ax="noplot")
# graphing.line_plot(indeg_x,indeg_y,c="k",style="x",label="",ax=axes[0])
# indegE_x,indegE_y = graphing.bar_chart_INT(net.in_degree_exc,c="r",ax="noplot")
# graphing.line_plot(indegE_x,indegE_y,c="r",style="x",label="exc",ax=axes[0])
# indegI_x,indegI_y = graphing.bar_chart_INT(net.in_degree_inh,c="b",ax="noplot")
# graphing.line_plot(indegI_x,indegI_y,c="b",style="x",label="inh",ax=axes[0])
# outdeg_x,outdeg_y = graphing.bar_chart_INT(net.out_degree,c="k",ax="noplot")
# graphing.line_plot(outdeg_x,outdeg_y,c="k",style="x",label="",ax=axes[1])
# axes[0].set(xlabel="incoming degree",ylabel="number of occurrence")
# axes[1].set(xlabel="outgoing degree")
# axes[0].set(xscale="log",yscale="log")
# axes[1].set(xscale="log",yscale="log")
# axes[0].legend()

# plt.tight_layout()
# # plt.show()
# fig.savefig("fig_netSF_degdist",dpi=150)
