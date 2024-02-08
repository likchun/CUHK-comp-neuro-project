from mylib3 import *
from scipy import stats
graphing.modern_style()
graphing.default_legend_style()


# net = NNetwork()
# net.adjacency_matrix_from_file("/Users/likchun/NeuroProject/networks/matrixB.txt")

# fig,axes = plt.subplots(1,2,figsize=(12,6))
# graphing.bar_chart_INT(net.in_degree_exc,c="r",label="exc",ax=axes[0])
# graphing.bar_chart_INT(net.in_degree_inh,c="b",label="inh",ax=axes[0])
# outdeg_x,outdeg_y = graphing.bar_chart_INT(net.out_degree,c="k",ax=axes[1])
# m_fit,s_fit = stats.norm.fit(net.out_degree)
# print(m_fit,s_fit)
# gausfit_x = np.linspace(0,25,100)
# gausfit_y = stats.norm.pdf(gausfit_x,m_fit,s_fit)
# graphing.line_plot(gausfit_x,gausfit_y*1000,c="g",style="--",ax=axes[1])
# axes[0].set(xlabel="incoming degree",ylabel="number of occurrence")
# axes[1].set(xlabel="outgoing degree")
# axes[0].set(xlim=(0,10),ylim=(0,None))
# axes[1].set(xlim=(0,23),ylim=(0,None))
# axes[0].legend()

# plt.tight_layout()
# plt.show()
# fig.savefig("fig_netB_degdist",dpi=150)


net = NNetwork()
net.adjacency_matrix_from_file("/Users/likchun/NeuroProject/networks/matrixSF.txt")

fig,axes = plt.subplots(1,2,figsize=(12,6))
indeg_x,indeg_y = graphing.bar_chart_INT(net.in_degree,c="k",ax="noplot")
graphing.line_plot(indeg_x,indeg_y,c="k",style="x",label="",ax=axes[0])
indegE_x,indegE_y = graphing.bar_chart_INT(net.in_degree_exc,c="r",ax="noplot")
graphing.line_plot(indegE_x,indegE_y,c="r",style="x",label="exc",ax=axes[0])
indegI_x,indegI_y = graphing.bar_chart_INT(net.in_degree_inh,c="b",ax="noplot")
graphing.line_plot(indegI_x,indegI_y,c="b",style="x",label="inh",ax=axes[0])
outdeg_x,outdeg_y = graphing.bar_chart_INT(net.out_degree,c="k",ax="noplot")
graphing.line_plot(outdeg_x,outdeg_y,c="k",style="x",label="",ax=axes[1])
axes[0].set(xlabel="incoming degree",ylabel="number of occurrence")
axes[1].set(xlabel="outgoing degree")
axes[0].set(xscale="log",yscale="log")
axes[1].set(xscale="log",yscale="log")
axes[0].legend()

plt.tight_layout()
# plt.show()
fig.savefig("fig_netSF_degdist",dpi=150)
