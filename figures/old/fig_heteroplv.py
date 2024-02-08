from neurosynlib import *
import itertools
graphing.modern_style()
graphing.default_legend_style()


directory0 = "/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"
# data_path = "/Users/likchun/NeuroProject/in_progress/202309A/scalefree_net_wi0.2-we0.8_a5_seed1"


w_inh = .2
w_exc = 1
alpha = 5

data_path = directory0+"scalefree/scalefree_net_wi{0}-we{1}_a{2}".format(w_inh,w_exc,alpha)
sd = SimulationData(data_path)
(gplv,pairwise_plvs,_) = plvtool.load_plv_data(data_path+"/plv_data")

ns = NeuroSynch()
ns.load_plvdata(pairwise_plvs)
ns.begin_clustering(transform_fn="contrast",_a=8)
ns.find_clusters(num_clusters=2)
ns.calculate_coherent_plv()
ns.print_info()


### figures ###

# fig_plvmat_orig,ax_plvmat_orig = plt.subplots(figsize=(7,6))
# ns.draw_original_plv_matrix(ax=ax_plvmat_orig)
# plt.tight_layout()

fig_plvmat_clust,ax_plvmat_clust = plt.subplots(figsize=(7,6))
ns.draw_clustered_plv_matrix(ax=ax_plvmat_clust)
plt.tight_layout()

fig_plvdist_clust,ax_plvdist_clust = plt.subplots(figsize=(7,6))
xys = ns.draw_clustered_plvs_distribution(sharebins=True,ax=ax_plvdist_clust)
plt.tight_layout()

# fig_plvcumul_clust,ax_plvcumul_clust = plt.subplots(figsize=(7,6))
# ns.draw_clustered_plvs_distribution_cumul(ax=ax_plvcumul_clust)
# plt.tight_layout()

# fig_raster_clust,ax_raster_clust = plt.subplots(figsize=(12,6))
# ns.draw_clustered_raster_plot(sd.dynamics.spike_times,ax=ax_raster_clust)
# plt.tight_layout()

# fig_isidist_clust,ax_isidist_clust = plt.subplots(figsize=(7,6))
# plotdata = ns.draw_clustered_isi_distributions(np.hstack(sd.dynamics.interspike_intervals),binsize=.08,ax="noplot")
# graphing.line_plot(*plotdata[0],style="--",ax=ax_isidist_clust)
# graphing.line_plot(*plotdata[1],ax=ax_isidist_clust)
# ax.set_xscale("log")
# plt.show()

# fig_dendrogram,ax_dendrogram = plt.subplots(figsize=(12,6))
# ns.draw_dendrogram(ax=ax_dendrogram)
# plt.tight_layout()



def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)




def overlap_coeff(probability_densdist1,probability_densdist2):
    """The two distributions must share the same set of discretized bins.\n
    probability_dendist = ["x values of the probability density distribution", "y values"]."""
    probability_densdist1 = np.array(probability_densdist1)
    probability_densdist2 = np.array(probability_densdist2)
    pdd1 = probability_densdist1[:,first_nonzero(probability_densdist1[1],0):last_nonzero(probability_densdist1[1],0)]
    pdd2 = probability_densdist2[:,first_nonzero(probability_densdist2[1],0):last_nonzero(probability_densdist2[1],0)]
    overlap_range = (np.max((pdd1[0].min(),pdd2[0].min())),np.min((pdd1[0].max(),pdd2[0].max())))
    pdd1_idxrange = (int(np.argwhere(pdd1[0]==overlap_range[0])),int(np.argwhere(pdd1[0]==overlap_range[1])))
    pdd2_idxrange = (int(np.argwhere(pdd2[0]==overlap_range[0])),int(np.argwhere(pdd2[0]==overlap_range[1])))
    share_pdd1 = pdd1[:,pdd1_idxrange[0]:pdd1_idxrange[1]]
    share_pdd2 = pdd2[:,pdd2_idxrange[0]:pdd2_idxrange[1]]
    overlap_area_sum = np.sum(np.minimum(share_pdd1[1],share_pdd2[1]))*(share_pdd1[0][1]-share_pdd1[0][0])
    return overlap_area_sum


print(overlap_coeff(xys[0],xys[1]))






# w_inh = .2
# w_exc = [.2,1]
# alpha = 5

# fig,axes = plt.subplots(4,3,figsize=(12,8),sharex="col")

# for i, w_e in enumerate(w_exc):
#     directory = directory0+"scalefree/scalefree_net_wi{0}-we{1}_a{2}".format(w_inh,w_e,alpha)
#     sd = SimulationData(directory)
#     (gplv,pairwise_plvs,_) = plvtool.load_plv_data(directory+"/plv_data")

#     ns = NeuroSynch()
#     ns.load_plvdata(pairwise_plvs)
#     ns.begin_clustering(transform_fn="contrast",_a=8.)
#     ns.find_clusters(num_clusters=2)
#     ns.calculate_coherent_plv()
#     ns.print_info()

#     ns.draw_original_plv_matrix(show_colorbar=False,ax=axes[i][0])





plt.show()

# ns.save_info("info_clusters_SFwe1_fn1-x")
# fig_plvmat_orig.savefig("fig_plvmatrix0",dpi=300)
# fig1.savefig("fig_plvmatrix_SFwe1_fn1-x",dpi=300)
# fig2.savefig("fig_plvdist_SFwe1_fn1-x",dpi=300)
# fig3.savefig("fig_rasterplot_SFwe1_fn1-x",dpi=300)
# # fig4.savefig("fig_dendrogram",dpi=300)
