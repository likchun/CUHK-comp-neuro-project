import os, sys
from libs.mylib3 import SimulationData, load_time_series
from libs.plvlib import PLVtool, Compressor
plvtool = PLVtool()
try: data_directory = str(sys.argv[1])
except IndexError: print("missing required argument: \"data_directory\"\n> for example: /Users/data/output"); exit(1)

################################################################################

# data_directory = "/Users/likchun/NeuroProject/..."
highfreqcut = 500 # same high-freq cutoff for all cases

################################################################################


sd = SimulationData(data_directory)

# cmpr = Compressor()
# filename = "memp_uint8" # use the compressed membrane potentials
# signals = cmpr.decode_uintX_t(*cmpr.load_encoded(os.path.join(data_directory,filename)))

filename = "memp.bin" # use the original uncompressed membrane potentials
signals = load_time_series(os.path.join(data_directory,filename),sd.settings["num_neuron"])

plvtool.configure_discard_times(discard_dynamics_time_ms=1000) # discard transient time
plvtool.configure_bandpass_filter(highcut=highfreqcut)
plvtool.import_data(signals,sd.settings["stepsize_ms"])
(globalplv,pairwiseplv) = plvtool.compute_plv_cuda()
# (globalplv,pairwiseplv) = plvtool.compute_plv()

plvtool.save_plv_data(os.path.join(data_directory,"plv_data"))