from preamble import *
from import_NetworkBursts import get_network_burst_stat


gIs = [0,.2,.4,.6,1,2,4,6,8]
params = [(0.2,gI,3,1) for gI in gIs]
binsize_ms = 5
bin_l_ms = 35 # should be integral multiple of binsize_ms
bin_r_ms = 65 # should be integral multiple of binsize_ms

nds = [NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*par))) for par in params]
[nd.remove_dynamics(500,0) for nd in nds]

network_burst_stat = get_network_burst_stat(nds)
np.save("data_netburstInh,gE02.npy", network_burst_stat)

print(network_burst_stat)