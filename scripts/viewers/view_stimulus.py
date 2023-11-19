from libs.mylib3 import load_stimulus, ts, graphing
from matplotlib import pyplot as plt
import sys
try: file_path = str(sys.argv[1])
except IndexError: print("missing required argument: \"file_path\"\n> for example: /Users/data/stim_file"); exit(1)

################################################################################

# file_path = "/Users/likchun/NeuroProject/..."

################################################################################

I_exc_thresh = 3.774
I_inh_thresh = 3.859

stim_Nidx, stim_series,stim_info = load_stimulus(file_path,returnInfo=True)


fig,ax = plt.subplots(figsize=(10,4))
totalduration = stim_info[1]+stim_info[2]+stim_info[3]
timestepsize = stim_info[0]
graphing.line_plot(ts(totalduration,timestepsize)/1000,stim_series,c="k",label="",ax=ax)
# graphing.line_plot([0,5],[I_exc_thresh,I_exc_thresh],c="m",style="--",lw=1,label="",ax=ax1)
# graphing.line_plot([0,5],[I_inh_thresh,I_inh_thresh],c="c",style="--",lw=1,label="",ax=ax1)
ax.set(xlim=(0,totalduration/1000))
ax.set(xlabel="time (s)",ylabel="")
plt.tight_layout()
plt.show()
# fig.savefig("stimulus.png")
