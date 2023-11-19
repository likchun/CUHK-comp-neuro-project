"""
NeuroSynlib
-----------

Last update: 5 September 2023
"""


from mylib3 import graphing, fill_lower_trimatrix
from plvlib import PLVtool
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial import distance
from matplotlib import pyplot as plt
import numpy as np

plvtool = PLVtool()


class NeuroSynch:

    def __init__(self) -> None:
        self._loadplvdatadone = False
        self._clusteringdone = False
        self._findclusterdone = False

    def load_plvdata(self,pairwise_plvs):
        """`pairwise_plvs`: a condensed NC2 array from a NxN square matrix"""
        self.plv_matrix = fill_lower_trimatrix(pairwise_plvs)+fill_lower_trimatrix(pairwise_plvs).T
        self._num_neuron = self.plv_matrix.shape[0]
        self.plv_matrix += np.eye(self._num_neuron)
        self._loadplvdatadone = True

    def begin_clustering(self,transform_fn="contrast",_a=10.):
        """
        `transform_fn`: function to transform PLVs to distances D\n
        - `"linear"`: D = 1 - PLV
        - `"concave"`: D = sqrt( 1 - PLV^2 ), lenient towards low PLV
        - `"convex"`: D = 1 - sqrt( 1 - (PLV-1)^2 ), lenient towards high PLV
        - `"contrast"`: D = { 1/(A-1) * [ A^2 exp(-a*PLV) - 1 ]/[ A exp(-a*PLV) + 1 ], A = exp(a/2) for a>0; 1 - PLV for a=0
            - `_a`: the parameter a for "constrast"
        """
        self._check_loadplvdatadone()
        if transform_fn=="linear": distance_matrix = 1-self.plv_matrix # linearly inversely proportional
        elif transform_fn=="concave": distance_matrix = np.sqrt(1-np.power(self.plv_matrix,2)) # concave, lenient towards low PLV
        elif transform_fn=="convex": distance_matrix = -np.sqrt(1-np.power(self.plv_matrix-1,2))+1 # convex, lenient towards high PLV
        elif transform_fn=="contrast":
            A = np.exp(_a/2)
            if _a==0: distance_matrix = 1-self.plv_matrix
            elif _a>0: distance_matrix = (A*A*np.exp(-_a*self.plv_matrix)-1)/(A*np.exp(-_a*self.plv_matrix)+1)/(A-1)
            else: raise ValueError("invalid value for the argument \"_a\"")
            np.fill_diagonal(distance_matrix,0)
        else: raise ValueError("invalid value for the argument \"transform_fn\"")
        self._linkage_data = linkage(distance.squareform(distance_matrix),method="ward",metric="euclidean")
        self._clusteringdone = True

    def find_clusters(self,num_clusters=2):
        self._check_clusteringdone()
        self.num_clusters = num_clusters
        self._labels = fcluster(self._linkage_data,self.num_clusters,criterion="maxclust")-1
        cluster_meanplv = np.array([np.mean(self.plv_matrix[np.ix_(np.argwhere(self._labels==i).flatten(),np.argwhere(self._labels==i).flatten())]) for i in range(self.num_clusters)])
        self.cluster_meanplv = cluster_meanplv[np.argsort(cluster_meanplv)[::-1]]
        self._nidx_mostsync = np.argwhere(self._labels==np.argsort(cluster_meanplv)[::-1][0]).flatten()
        self.cluster_indices = [np.argwhere(self._labels==i).flatten() for i in np.arange(self.num_clusters)[np.argsort(cluster_meanplv)[::-1]]]
        self._cluster_indices_allsort = np.concatenate([np.argwhere(self._labels==i).flatten() for i in np.arange(self.num_clusters)[np.argsort(cluster_meanplv)[::-1]]])
        self.cluster_size = np.array([len(np.argwhere(self._labels==i).flatten()) for i in np.arange(self.num_clusters)[np.argsort(cluster_meanplv)[::-1]]],dtype=int)
        self.plv_matrix_allsort = self.plv_matrix[np.ix_(self._cluster_indices_allsort,self._cluster_indices_allsort)]
        self.fraction_of_coherence = self.cluster_size[0]/self._num_neuron
        self._findclusterdone = True

    def draw_dendrogram(self,ax=None):
        self._check_clusteringdone()
        if ax==None: fig,ax = plt.subplots(figsize=(12,6))
        dendrogram(self._linkage_data,ax=ax)
        ax.set(ylabel="dissimilarity/distance",xlabel="neuron index")

    def draw_original_plv_matrix(self,ax=None):
        self._check_loadplvdatadone()
        if ax==None: fig,ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(self.plv_matrix,vmin=0,vmax=1,cmap="viridis")
        ax.set(xticks=[1]+[self._num_neuron],yticks=[1]+[self._num_neuron])
        ax.set(xlim=(1,self._num_neuron),ylim=(1,self._num_neuron))
        ax.set(xlabel="neuron")
        ax.grid(False)
        plt.colorbar(im,ax=ax,label="pair-wise PLV",shrink=.8,aspect=30*.8)

    def draw_clustered_plv_matrix(self,ax=None,cluster_separation_lines="red"):
        """`cluster_separation_lines`: str, colour of the separation line, `"none"` to hide it"""
        self._check_clusteringdone()
        self._check_findclusterdone()
        if ax==None: fig,ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(self.plv_matrix_allsort,vmin=0,vmax=1,cmap="viridis")
        if cluster_separation_lines=="none": pass
        else: [[graphing.line_plot([1,self._num_neuron],[self.cluster_size[:i].sum(),self.cluster_size[:i].sum()],c=cluster_separation_lines,style="--",lw=1,ax=ax),graphing.line_plot([self.cluster_size[:i].sum(),self.cluster_size[:i].sum()],[1,self._num_neuron],c=cluster_separation_lines,style="--",lw=1,ax=ax)] for i in range(1,self.num_clusters)]
        ax.set(xticks=[1]+[self.cluster_size[:i].sum() for i in range(self.num_clusters)]+[self._num_neuron],yticks=[1]+[self.cluster_size[:i].sum() for i in range(self.num_clusters)]+[self._num_neuron])
        ax.set(xlim=(1,self._num_neuron),ylim=(1,self._num_neuron))
        ax.set(xlabel="neuron (clustered)")
        ax.grid(False)
        plt.colorbar(im,ax=ax,label="pair-wise PLV",shrink=.8,aspect=30*.8)

    def draw_clustered_raster_plot(self,spike_times,ax=None,cluster_separation_lines="red"):
        """`cluster_separation_lines`: str, colour of the separation line, `"none"` to hide it"""
        self._check_clusteringdone()
        self._check_findclusterdone()
        if ax==None: fig,ax = plt.subplots(figsize=(12,6))
        graphing.raster_plot_network(spike_times[self._cluster_indices_allsort],ax=ax)
        if cluster_separation_lines=="none": pass
        else: [[graphing.line_plot([np.amin(np.hstack(spike_times))/1000,np.amax(np.hstack(spike_times))/1000],[self.cluster_size[:i].sum(),self.cluster_size[:i].sum()],c=cluster_separation_lines,style="--",lw=1,ax=ax),graphing.line_plot([self.cluster_size[:i].sum(),self.cluster_size[:i].sum()],[np.amin(np.hstack(spike_times))/1000,np.amax(np.hstack(spike_times))/1000],c=cluster_separation_lines,style="--",lw=1,ax=ax)] for i in range(1,self.num_clusters)]
        ax.set(yticks=[1]+[self.cluster_size[:i].sum() for i in range(self.num_clusters)]+[self._num_neuron])
        ax.set(xlim=(np.amin(np.hstack(spike_times))/1000,np.amax(np.hstack(spike_times))/1000),ylim=(1,self._num_neuron))
        ax.set(xlabel="time (s)",ylabel="neuron (clustered)")
        ax.grid(False)

    def draw_clustered_plvs_distribution(self,binsize=.01,ax=None):
        self._check_clusteringdone()
        self._check_findclusterdone()
        if ax==None: fig,ax = plt.subplots(figsize=(7,6))
        graphing.distribution_density_plot(self.plv_matrix_allsort[np.triu_indices(self._num_neuron,k=1)],binsize,c=graphing.mycolors[0],style=":",a=.5,lw=1.5,label="full",ax=ax)
        for i in range(self.num_clusters):
            if i==0:
                x,y = graphing.distribution_density_plot(self.plv_matrix_allsort[:self.cluster_size[i],:self.cluster_size[i]][np.triu_indices(self.cluster_size[i],k=1)],binsize,ax="noplot")
                graphing.line_plot(x,y*(self.cluster_size[i]/self._num_neuron)**0,c=graphing.mycolors[i+1],label="{}".format(i+1),ax=ax)
            else:
                x,y = graphing.distribution_density_plot(self.plv_matrix_allsort[self.cluster_size[:i].sum():self.cluster_size[:i+1].sum(),self.cluster_size[:i].sum():self.cluster_size[:i+1].sum()][np.triu_indices(self.cluster_size[i],k=1)],binsize,ax="noplot")
                graphing.line_plot(x,y*(self.cluster_size[i]/self._num_neuron)**0,c=graphing.mycolors[i+1],label="{}".format(i+1),ax=ax)
        ax.set(xlabel="pairwise PLV",ylabel="probability density")
        ax.legend(title="cluster#")
        ax.set(xlim=(0,1))
        ax.grid(False)

    def calculate_effective_plv(self):
        self._check_clusteringdone()
        self._check_findclusterdone()
        self.global_plv = np.sum(np.tril(self.plv_matrix))*2/(self._num_neuron*(self._num_neuron-1))
        self.effective_plv = np.sum(np.tril(self.plv_matrix[np.ix_(self._nidx_mostsync,self._nidx_mostsync)]))*2/(self.cluster_size[0]*(self.cluster_size[0]-1))
        return self.effective_plv

    def print_info(self):
        self._check_clusteringdone()
        self._check_findclusterdone()
        print("global PLV: {:.5f}".format(self.global_plv))
        print("effective PLV: {:.5f}".format(self.effective_plv))
        print("cluster PLVs: {:.5f}, {:.5f}".format(*self.cluster_meanplv))
        print("cluster size: {:d}, {:d}".format(*self.cluster_size))
        print("fraction of coherence: {:.3f}".format(self.fraction_of_coherence))

    def save_info(self,filename):
        self._check_clusteringdone()
        self._check_findclusterdone()
        with open("./"+filename+".txt","w") as f:
            f.write("global PLV: {:.5f}\n".format(self.global_plv))
            f.write("effective PLV: {:.5f}\n".format(self.effective_plv))
            f.write("cluster PLVs: {:.5f}, {:.5f}\n".format(*self.cluster_meanplv))
            f.write("cluster size: {:d}, {:d}\n".format(*self.cluster_size))
            f.write("fraction of coherence: {:.3f}\n".format(self.fraction_of_coherence))

    def _check_loadplvdatadone(self):
        if not self._loadplvdatadone: raise Exception("call the function `load_plvdata()` before this")

    def _check_clusteringdone(self):
        if not self._clusteringdone: raise Exception("call the function `begin_hierarchical_clustering()` before this")

    def _check_findclusterdone(self):
        if not self._findclusterdone: raise Exception("call the function `find_clusters()` before this")
