# Last update: 17 Nov 2023

######################
### Basic settings ###
######################

stim_name = "sinwave_nonneg"
stim_id = "stim_p1c"
fd_Hz = 7 # driving frequency
A = 48 # stimulus amplitude
stim_neuron_frac = 1 # the fraction of all neurons that are subject to the stimulus

# ensure the following values matche the numerical simulation settings #
stim_duration_ms = 4000
nonstim_duration_beg_ms = 1000
nonstim_duration_end_ms = 0
dt_ms = .01

######################

import sys, os
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir,".."))
from libs.mylib3 import *
random.seed(0)
np.random.seed(0)

if nonstim_duration_end_ms != 0:
    if np.mod(stim_duration_ms,1000/fd_Hz) == 0: _endtime_fix = 0
    else: _endtime_fix = int(1000/fd_Hz-np.mod(stim_duration_ms,1000/fd_Hz))
    stim_duration_ms += _endtime_fix
    nonstim_duration_end_ms -= _endtime_fix
else: pass

# introduce a phase shift such that the stimulus is continuous at the onset
# effectively, the stimulus S(t) = Asin(...) becomes: S(t) = -Acos(...)
phase_shift = -np.pi/2
stimulus_time_series_periodic = np.array([1*np.sin(2*np.pi*fd_Hz/1000*t+phase_shift) for t in ts(stim_duration_ms,dt_ms)])

B = A # non-negative S(t)
stimulus = A * stimulus_time_series_periodic + B # general form of S(t)
stimulus = np.pad(stimulus,(int(nonstim_duration_beg_ms/dt_ms),int(nonstim_duration_end_ms/dt_ms)),"constant")

stim_nidx_exc = np.random.choice(range(200,1000),int(800*stim_neuron_frac),replace=False).astype(int)
stim_nidx_inh = np.random.choice(range(0,200),int(200*stim_neuron_frac),replace=False).astype(int)
stim_nidx_tot = np.concatenate((stim_nidx_exc,stim_nidx_inh)).astype(int)
stim_nidx = stim_nidx_tot # change this variable to stimulate only exc, only inh, or both


print("stimulus name/type: {}".format(stim_name))
print("amplitude: {}".format(A))
print("driving frequency: {} Hz".format(fd_Hz))
print("total duration: {} ms".format(nonstim_duration_beg_ms+stim_duration_ms+nonstim_duration_end_ms))
print("> before simluation duration: {} ms".format(nonstim_duration_beg_ms))
print("> stimulation duration: {} ms".format(stim_duration_ms))
print("> after simluation duration: {} ms".format(nonstim_duration_end_ms))
print("total number of stimulated neurons: {}".format(len(stim_nidx)))
print("fraction of stimulated neurons: {}".format(stim_neuron_frac))


fig, ax = plt.subplots(figsize=(7,3))
ax.plot(np.arange(len(stimulus))*dt_ms/1000,stimulus,c="k")
ax.set(xlabel="time (s)",ylabel="")
ax.set(xlim=(0,(nonstim_duration_beg_ms+stim_duration_ms+nonstim_duration_end_ms)/1000))
ax.grid(True)
plt.tight_layout()
plt.show()


outfilename = "{}[A{},f{},m{}].txt".format(stim_id,A,fd_Hz,stim_neuron_frac)
# Naming:
# "p" stands for periodic
# "1" indicates the specific type or form of the stimulus
# "a" differentiates the different settings of the same stimulus, e.g., dt, stim_duration, etc.
# the value following "A" corresponds to the stimulus amplitude
# the value following "f" is the stimulus driving frequency
# the value following "m" is the fraction of stimulated neurons

# exit(0)

infos = [dt_ms,stim_duration_ms,nonstim_duration_beg_ms,nonstim_duration_end_ms,stim_name,fd_Hz,A,B]
infos = np.array(infos,dtype=str)
with open(os.path.join(this_dir,outfilename),"w") as f:
    f.write("\t".join(infos))
    f.write("\n")
    f.write("\t".join(map(str,stim_nidx)))
    f.write("\n")
    # f.write("\t".join(map(str,stimulus)))
    f.write("\t".join(["{:.3f}".format(x) for x in stimulus]))
# File content:
# (1st line) information of the stimulus, see variable "infos" above
# (2nd line) the indices of stimulated neurons
# (3rd line) the stimulus time series, the n-th value is the stimulus magnitude at the n-th time step