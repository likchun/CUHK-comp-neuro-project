from preamble import *
from import_NetworkBursts import get_network_burst_stat


stimuli = Stimuli([20,40,60,80,100])

# param = (0.2, 0.6, 3, 1)
param = (0.2, 0.2, 3, 1)
# param = (0.3, 0.2, 3, 1)

nds = [
    NeuroData(os.path.join(data_path.SD_resptostim_const, "{},{},{},{}/[{},0,2,0]".format(*param, s)))
for s in [0]+stimuli.S]
[[nd.remove_dynamics(500,0),nd.apply_neuron_mask(stimuli.obsvgp_ind)] for nd in nds]

binsize_ms = 5
bin_l_ms = 35 # should be integral multiple of binsize_ms
bin_r_ms = 65 # should be integral multiple of binsize_ms

network_burst_stat = get_network_burst_stat(nds)

# np.save(os.path.join(data_path.this_dir, "data_netburststat_e02i06.npy"), network_burst_stat)
# np.save(os.path.join(data_path.this_dir, "data_netburststat_e02i02.npy"), network_burst_stat)
# np.save(os.path.join(data_path.this_dir, "data_netburststat_e03i02.npy"), network_burst_stat)

print(network_burst_stat)