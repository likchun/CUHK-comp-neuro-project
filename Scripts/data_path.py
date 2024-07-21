import os, sys

this_dir = sys.path[0]

_base_dir = ""

network_A = os.path.join(_base_dir, "Networks/network_A.txt")

_simdata_dir = os.path.join(_base_dir, "Simulation-data")

SD_netA_mapout_gEgI = os.path.join(_simdata_dir, "network-A/mapout_gEgI")
SD_netA_mapout_adap = os.path.join(_simdata_dir, "network-A/mapout_adap")
SD_netA_timeseries  = os.path.join(_simdata_dir, "network-A/timeseries")
SD_netA_trans_drive = os.path.join(_simdata_dir, "network-A/transient_drive")
SD_netB_mapout_gEgI = os.path.join(_simdata_dir, "network-B/width0.08,a5_EMdt02")

SD_isolated_const   = os.path.join(_simdata_dir, "isolated_const-current")
SD_isolated_stoch_e = os.path.join(_simdata_dir, "isolated_stoch-current/exc")
SD_isolated_stoch_i = os.path.join(_simdata_dir, "isolated_stoch-current/inh")

SD_resptostim_step  = os.path.join(_simdata_dir, "responses-to-stimuli/step-stim-60")
SD_resptostim_const = os.path.join(_simdata_dir, "responses-to-stimuli/const-stim")

SD_state1_pXY       = os.path.join(_simdata_dir, "state1-pXY")

_analysis_dir = os.path.join(_base_dir, "Data-analysis")

DA_phasediag        = os.path.join(_analysis_dir, "phase-diagrams")
DA_resptostim       = os.path.join(_analysis_dir, "responses-to-stimuli")
DA_state2_netburst  = os.path.join(_analysis_dir, "state2-netburst")
DA_state1_pXY       = os.path.join(_analysis_dir, "state1-pXY")
