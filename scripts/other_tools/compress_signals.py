import os, sys
from libs.mylib3 import SimulationData, Compressor, load_time_series
cmpr = Compressor()
try: data_directory = str(sys.argv[1])
except IndexError: print("missing required argument: \"data_directory\"\n> for example: /Users/data/output"); exit(1)

################################################################################

# data_directory = "/Users/likchun/NeuroProject/..."
duration_discarded = 500 # unit in ms
remove_original = False

################################################################################


sd = SimulationData(data_directory)

memp = "memp"
recv = "recv"
curr = "curr"
gcde = "gcde"
gcdi = "gcdi"
stoc = "stoc"

data_files = [memp,recv,curr,gcde,gcdi,stoc]

for f in data_files:
    try:
        cmpr.save_encoded(data_directory+"/"+f+"_uint8",*cmpr.encode_uintX_t(load_time_series(data_directory+"/"+f+".bin", neuron_num=sd.settings["neuron_num"])[:,int(duration_discarded/sd.settings["dt_ms"]):]))
        if remove_original:
            if os.path.isfile(os.path.join(data_directory,"memp.bin")):
                os.remove(os.path.join(data_directory,"memp.bin"))
                print("{} compressed, original removed".format(f))
        else: print("{} compressed, original NOT removed".format(f))
    except FileNotFoundError: print("{} does not exist, continue".format(f))

