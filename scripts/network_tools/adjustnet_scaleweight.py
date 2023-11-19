from libs.mylib3 import NNetwork, graphing
from matplotlib import pyplot as plt


def get_scaled_weight_net(base_network, w_exc, w_inh):
    net = NNetwork()
    net.adjacency_matrix_from_file("{}.txt".format(base_network))
    net.scale_synaptic_weights(w_exc, neuron_type="exc")
    net.scale_synaptic_weights(w_inh, neuron_type="inh")
    return net

if __name__=="__main__":
    # A base network contains only the structural information,
    # all exc synaptic weight = 1
    # all inh synaptic weight = -1
    base_network = "net_unifindeg"
    # this script scales the weights only
    w_inh = 0.2
    w_exc = 0.08

    net = get_scaled_weight_net(base_network, w_exc, w_inh)
    fig,ax = plt.subplots(figsize=(6,6))
    graphing.scatter_plot(*net.link_ij,ax=ax)
    ax.set(xlim=(0,1000),ylim=(0,1000))
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    # net.adjacency_matrix_to_file("{}_({},{}).txt".format(base_network,w_exc,w_inh))