# from mylib import NeuroData
from scipy import signal
from mylib import get_spike_count
import numpy as np


binsize_ms = 5
bin_l_ms = 35 # should be integral multiple of binsize_ms
bin_r_ms = 65 # should be integral multiple of binsize_ms


def get_network_burst_stat(nds):
    netburst_stat = []
    for i,nd in enumerate(nds):
        mean_fr = nd.dynamics.mean_firing_rate.mean()
        netfr = nd.dynamics.average_firing_rate_time_histogram(binsize_ms)
        pinds, props = signal.find_peaks(netfr[1], prominence=10)#, distance=100/binsize_ms)
        vinds = np.array((pinds[1:]+pinds[:-1])/2).astype(int)
        pinds = pinds[1:-1] # exclude the first and the last peaks
        pinds_instepsize = pinds * binsize_ms/nd.configs["stepsize_ms"]
        vinds_instepsize = vinds * binsize_ms/nd.configs["stepsize_ms"]

        netburst_frshapes = np.array([netfr[1][int(p-bin_l_ms/binsize_ms):int(p+bin_r_ms/binsize_ms)] for p in pinds])
        # populevent_frshape_avg = np.array([netfr[1][int(p-bin_l_ms/binsize_ms):int(p+bin_r_ms/binsize_ms)] for p in pinds]).mean(0)

        netburst_frshapes_norm = [shape-np.amin(shape) for shape in netburst_frshapes]
        netburst_frshapes_norm = np.array([shape/np.amax(shape) for shape in netburst_frshapes_norm])
        # populevent_frshapes_norm = populevent_frshapes/np.amax(populevent_frshape_avg)

        pwidths_l_03 = np.array([-np.interp(.3, frshape[:int(bin_l_ms/binsize_ms)+1], np.flip(np.arange(0,bin_l_ms+binsize_ms,binsize_ms))) for frshape in netburst_frshapes_norm])
        pwidths_r_03 = np.array([np.interp(.3, np.flip(frshape[int(bin_l_ms/binsize_ms):]), np.flip(np.arange(0,bin_r_ms,binsize_ms))) for frshape in netburst_frshapes_norm])
        pwidths_03   = pwidths_r_03 - pwidths_l_03
        pwidths_r_04 = np.array([np.interp(.4, np.flip(frshape[int(bin_l_ms/binsize_ms):]), np.flip(np.arange(0,bin_r_ms,binsize_ms))) for frshape in netburst_frshapes_norm])
        pwidths_l_04 = np.array([-np.interp(.4, frshape[:int(bin_l_ms/binsize_ms)+1], np.flip(np.arange(0,bin_l_ms+binsize_ms,binsize_ms))) for frshape in netburst_frshapes_norm])
        pwidths_04   = pwidths_r_04 - pwidths_l_04
        pwidths_l_05 = np.array([-np.interp(.5, frshape[:int(bin_l_ms/binsize_ms)+1], np.flip(np.arange(0,bin_l_ms+binsize_ms,binsize_ms))) for frshape in netburst_frshapes_norm])
        pwidths_r_05 = np.array([np.interp(.5, np.flip(frshape[int(bin_l_ms/binsize_ms):]), np.flip(np.arange(0,bin_r_ms,binsize_ms))) for frshape in netburst_frshapes_norm])
        pwidths_05   = pwidths_r_05 - pwidths_l_05

        splitinds = vinds_instepsize
        # splitinds = np.sort(np.concatenate([[0],pinds_instepsize,vinds_instepsize,[nd.configs["num_step"]]])).flatten()
        splited_spikes = [[
            spike_steps[np.argwhere((splitinds[i] < spike_steps) & (spike_steps <= splitinds[i+1]))]
                for spike_steps in nd.dynamics.spike_steps]
                    for i in range(len(splitinds)-1)]

        # fig,ax=plt.subplots()
        # ax.plot((np.arange(len(netburst_frshape_norm))-int(bin_l_ms/binsize_ms))*binsize_ms, netburst_frshape_norm, "kx-")
        # ax.plot(pwidth_l, .3, "gs", ms=10)
        # ax.plot(pwidth_r, .3, "ms", ms=10)
        # plt.show()

        # fig,ax=plt.subplots()
        # qgraph.raster_plot(nd.dynamics.spike_times, ax=ax, colors="k", marker=".", ms=3, mec="none", time_scale="ms")
        # [ax.plot(np.full(2,pind*nd.configs["stepsize_ms"]), [0,1000], "r--", lw=1) for pind in pinds]
        # [ax.plot(np.full(2,vind*nd.configs["stepsize_ms"]), [0,1000], "g--", lw=1) for vind in vinds]
        # plt.show()

        num_netburst = len(pinds)
        if num_netburst != len(splited_spikes): raise ValueError("num_peaks does not match len(splited_spikes)")
        freq_nb = num_netburst/nd.configs["duration_ms"]*1000
        INBI_ms = np.diff(pinds_instepsize*nd.configs["stepsize_ms"])
        mean_INBI_ms = np.mean(INBI_ms)
        std_INBI_ms = np.std(INBI_ms)

        K_nb = np.array([get_spike_count(ss).sum() for ss in splited_spikes])
        mean_K_nb = np.mean(K_nb)
        std_K_nb = np.std(K_nb)

        mean_T_nb_03 = np.mean(pwidths_03)
        std_T_nb_03 = np.std(pwidths_03)
        mean_T_nb_04 = np.mean(pwidths_04)
        std_T_nb_04 = np.std(pwidths_04)
        mean_T_nb_05 = np.mean(pwidths_05)
        std_T_nb_05 = np.std(pwidths_05)

        Nactive_nb_0 = np.array([len(get_spike_count(ss)[get_spike_count(ss)>0]) for ss in splited_spikes])
        Nactive_nb_1 = np.array([len(get_spike_count(ss)[get_spike_count(ss)>1]) for ss in splited_spikes])
        Nactive_nb_2 = np.array([len(get_spike_count(ss)[get_spike_count(ss)>2]) for ss in splited_spikes])
        mean_Nactive_nb_0 = np.mean(Nactive_nb_0)
        std_Nactive_nb_0 = np.std(Nactive_nb_0)
        mean_Nactive_nb_1 = np.mean(Nactive_nb_1)
        std_Nactive_nb_1 = np.std(Nactive_nb_1)
        mean_Nactive_nb_2 = np.mean(Nactive_nb_2)
        std_Nactive_nb_2 = np.std(Nactive_nb_2)

        netburst_stat.append([
            mean_fr,
            num_netburst,
            freq_nb,
            mean_INBI_ms,
            std_INBI_ms,
            mean_K_nb,
            std_K_nb,
            mean_Nactive_nb_0,
            std_Nactive_nb_0,
            mean_Nactive_nb_1,
            std_Nactive_nb_1,
            mean_Nactive_nb_2,
            std_Nactive_nb_2,
            mean_T_nb_03,
            std_T_nb_03,
            mean_T_nb_04,
            std_T_nb_04,
            mean_T_nb_05,
            std_T_nb_05,
        ])
    return np.array(netburst_stat).T