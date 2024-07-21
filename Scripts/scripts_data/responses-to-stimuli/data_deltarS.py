from preamble import *

noiseLv = 3

params1 = [
    (0.6,  0.2, noiseLv, 1),
    (0.6,  0.6, noiseLv, 1),
    (0.3,  0.2, noiseLv, 1),
    (0.3,  0.6, noiseLv, 1),
    (0.2,  0.2, noiseLv, 1),
    (0.2,  0.6, noiseLv, 1),
    (0.04, 0.2, noiseLv, 1),
    (0.04, 0.6, noiseLv, 1),
]

stimuli = Stimuli([20,40,60,80,100])


if __name__=="__main__":
    ndss = [[
        NeuroData(os.path.join(data_path.SD_resptostim_const, "{},{},{},{}/[{},0,2,0]".format(*par, s)))
        for s in [0]+stimuli.S
    ] for par in params1]
    [[nd.remove_dynamics(500,0) for nd in nds] for nds in ndss]
    # consider only the group of neurons that never receive stimulation
    [[nd.apply_neuron_mask(stimuli.obsvgp_ind) for nd in nds] for nds in ndss]
    frs = [[nd.dynamics.mean_firing_rate.mean() for nd in nds] for nds in ndss]
    frs = np.array(frs)
    print(frs.astype(int))
    frs_chg = np.array([fr - fr[0] for fr in frs])
    print(frs_chg.astype(int))
    np.save(os.path.join(data_path.this_dir, "data_deltarS.npy"), frs)