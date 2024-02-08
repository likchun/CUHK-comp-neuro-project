from libs.mylib3 import NNetwork
import os, itertools


################
### Settings ###
################

base_network = "net_unifindeg"

stimulus_folder = "sinwave_nonneg"#"none"
stimulus_file = stimulus_folder+"/stim_p1a[A4,f5,m1].txt"#"none"

configs = [
    # (w_exc, w_inh, alpha, duration_ms, timestep_size_ms)
    # (0.02,0.2,3,5000,.02),
    # (0.04,0.2,3,5000,.02),
    # (0.05,0.2,3,5000,.02),
    # (0.06,0.2,3,5000,.02),
    # (0.07,0.2,3,5000,.02),
    # (0.08,0.2,3,5000,.02),
    # (0.12,0.2,3,5000,.02),
    # (0.2,0.2,3,5000,.02),
    # (0.3,0.2,3,5000,.05),
    # (0.4,0.2,3,5000,.02),
    # (0.5,0.2,3,5000,.02),
    # (0.6,0.2,3,5000,.02),
]

voltage_out = "false"
write_command = True # for Windows system
write_shell = False # for Linux systems

MAX_RESOURCE_CORES = 28

#################################################################


# if len(w_inh)*len(w_exc)*len(alphas) > MAX_RESOURCE_CORES: raise Exception("{} jobs exceeds MAX_RESOURCE_CORES={}".format(len(w_inh)*len(w_exc)*len(alphas), MAX_RESOURCE_CORES))

net = NNetwork()


config_txt = lambda netname, duration_ms, timestep_size_ms, alpha: """[Caution: do not modify anything except the values after the equal signs]

[Network and synaptic weights]
synaptic weights matrix filename    = {}.txt
matrix format (full/nonzero)#       = nonzero
file delimiter (tab/space/...)      = space
weights multiplying factor, beta*   = 1

[Numerical settings]
simulation duration, T (ms)         = {}
time step size, dt (ms)             = {}

[Stochastic current]
random number generation seed       = 0
white noise strength, alpha         = {}

[Constant current]
constant driving current            = 0

[Stimulus]
stimulus time series filename       = {}

[Initial values]
membrane potential (mV)             = -70
recovery variable                   = -14

[Spike history truncation time]
inhibitory neurons (ms)             = 600
excitatory neurons (ms)             = 500

[Time series exportation]
membrane potential (true/false)     = {}
recovery variable (true/false)      = false
presynaptic current (true/false)    = false
presynaptic EXC conductance         = false
presynaptic INH conductance         = false
stochastic current                  = false


*the synaptic weights are rescaled after finding in-degree and types of neurons

#full:      the data file stores a full size NxN matrix
#nonzero:   the data file stores only the nonzero elements with format (j i gji)""".format(netname,duration_ms,timestep_size_ms,alpha,stimulus_file,voltage_out)

if write_command:
    f_command = open("auto_execute.command", "w")
    f_command.write("echo [{} tasks in queue]\n".format(len(configs)))

foldernames = ""

for we, wi, a, T, dt in configs:
    if not we: print("w+:{0}, w-:{1}".format(we,wi))
    else: print("w+:{0}, w-:{1}, ratio:{2}".format(we,wi,wi/we))
    # netname = "{}_[{},{}]".format(base_network,we,wi)
    # foldername = "{}_[{},{},{}]".format(base_network,we,wi,a)
    netname = "{}_[{},{}]".format(base_network,we,wi)
    foldername = "{},{},{}".format(we,wi,a)
    foldernames += " \"{}\"".format(foldername)
    net.adjacency_matrix_from_file("{}.txt".format(base_network))
    net.scale_synaptic_weights(wi,neuron_type="inh")
    net.scale_synaptic_weights(we,neuron_type="exc")
    try: os.mkdir(foldername)
    except FileExistsError: raise
    net.adjacency_matrix_to_file(os.path.join(foldername, "{}.txt".format(netname)))
    f_config = open(os.path.join(foldername, "vars.txt"), "w")
    f_config.write(config_txt(netname,T,dt,a))
    if write_command:
        f_command.write("mv ./NetworkDynamics.o ./{}/NetworkDynamics.o\n".format(foldername))
        if stimulus_folder!="none": f_command.write("mv ./{1} ./{0}/{1}\n".format(foldername,stimulus_folder))
        f_command.write("cd {}\n".format(foldername))
        f_command.write("./NetworkDynamics.o -scm\n")
        f_command.write("mv ./NetworkDynamics.o ../NetworkDynamics.o\n")
        f_command.write("cd ./output\n")
        f_command.write("mv ./cont.dat ../cont.dat\n")
        f_command.write("mv ./info.txt ../info.txt\n")
        f_command.write("mv ./sett.json ../sett.json\n")
        f_command.write("mv ./spks.txt ../spks.txt\n")
        f_command.write("mv ./memp.bin ../memp.bin\n")
        f_command.write("cd ..\n")
        if stimulus_folder!="none": f_command.write("mv ./{0} ../{0}\n".format(stimulus_folder))
        f_command.write("rmdir ./output\n")
        f_command.write("cd ..\n".format(wi,we))

if write_command:
    f_command.write("echo [completed]\n")
    f_command.close()

foldernames = foldernames[1:]

bash_txt_1 = """
#!/bin/bash

#PBS -S /bin/bash               ## many Torque PBS directives can be found on internet
#PBS -o result.txt              ## (optional) std. output to myjob.out
#PBS -e error.txt               ## (optional) std. error to myjob.err
#PBS -l walltime=00:03:00       ## request max. 1 hour for running
#PBS -l nodes=1:ppn=28          ## run on 2 nodes and 28 processes per node
#PBS -q normal                  ## (optional) queue can be normal(default),bigmem,long,debug


cd $PBS_O_WORKDIR

echo "start: `date`"

sort < $PBS_NODEFILE | uniq -c          ## (optional) list which nodes used for this job

foldernames=({})

""".format(foldernames)
#PBS -M likchun@link.cuhk.edu.hk

if write_shell:
    f_shell = open("auto_execute.sh", "w")
    f_shell.write(bash_txt_1)
    f_shell.write(r"for fol in ${foldernames[@]} ; do (")
    f_shell.write("\n\tcp ./NetworkDynamics.o ./$fol/NetworkDynamics.o &&\n")
    f_shell.write("\tcd $fol &&\n")
    f_shell.write("\tchmod 777 ./NetworkDynamics.o &&\n")
    f_shell.write("\t./NetworkDynamics.o -scm &&\n")
    f_shell.write("\trm ./NetworkDynamics.o &&\n")
    f_shell.write("\tcd ./output &&\n")
    f_shell.write("\tmv ./cont.dat ../cont.dat &&\n")
    f_shell.write("\tmv ./info.txt ../info.txt &&\n")
    f_shell.write("\tmv ./sett.json ../sett.json &&\n")
    f_shell.write("\tmv ./spks.txt ../spks.txt &&\n")
    f_shell.write("\tmv ./memp.bin ../memp.bin &&\n")
    f_shell.write("\tcd .. &&\n")
    f_shell.write("\trmdir ./output &&\n")
    f_shell.write("\tcd ../) &\n")
    f_shell.write("done\n")
    f_shell.write("wait\n")
    f_shell.write("\necho \"end: `date`\"")