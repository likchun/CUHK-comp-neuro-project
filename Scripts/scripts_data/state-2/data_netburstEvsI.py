from preamble import *
from import_NetworkBursts import get_network_burst_stat


gEs = [.08,.1,.16,.2,.3,.36,.4]
params = [(gE,0.2,3,1) for gE in gEs]

# gEs = [.08,.1,.16,.2,.3,.36,.4]
# params = [(gE,0.6,3,1) for gE in gEs]

# gEs = [.08,.1,.16,.2,.3,.36,.4]
# params = [(gE,8,3,1) for gE in gEs]

binsize_ms = 5
bin_l_ms = 35 # should be integral multiple of binsize_ms
bin_r_ms = 65 # should be integral multiple of binsize_ms


nds = [NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*par))) for par in params]
[nd.remove_dynamics(500,0) for nd in nds]

network_burst_stat = get_network_burst_stat(nds)

# np.save("data_netburstExc,gI02.npy", network_burst_stat)
# np.save("data_netburstExc,gI06.npy", network_burst_stat)
# np.save("data_netburstExc,gI8.npy", network_burst_stat)

print(network_burst_stat)
