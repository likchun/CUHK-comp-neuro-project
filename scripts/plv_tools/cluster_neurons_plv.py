from libs.mylib3 import SimulationData
from libs.neurosynlib import *


data_path = "/Users/likchun/NeuroProject/..."


sd = SimulationData(data_path)
(gplv,pairwise_plvs,_) = plvtool.load_plv_data(data_path+"/plv_data")

ns = NeuroSynch()
ns.load_plvdata(pairwise_plvs)
ns.begin_hierarchical_clustering()
ns.find_clusters(num_clusters=2)
ns.calculate_effective_plv()
ns.print_info()


fig0,ax = plt.subplots(figsize=(7,6))
ns.draw_original_plv_matrix(ax=ax)
plt.tight_layout()

fig1,ax = plt.subplots(figsize=(7,6))
ns.draw_clustered_plv_matrix(ax=ax)
plt.tight_layout()

fig2,ax = plt.subplots(figsize=(7,6))
ns.draw_clustered_plvs_distribution(ax=ax)
plt.tight_layout()

fig3,ax = plt.subplots(figsize=(12,6))
ns.draw_clustered_raster_plot(sd.dynamics.spike_times,ax=ax)
plt.tight_layout()

fig4,ax = plt.subplots(figsize=(12,6))
ns.draw_dendrogram(ax=ax)
plt.tight_layout()


plt.show()

# ns.save_info("info_clusters")
# fig0.savefig("fig_plvmatrix0",dpi=300)
# fig2.savefig("fig_plvmatrix",dpi=300)
# fig4.savefig("fig_plvdist",dpi=300)
# fig3.savefig("fig_rasterplot",dpi=300)
# fig1.savefig("fig_dendrogram",dpi=300)
