from preamble import *


params = [
    (0.6,  0.2, 3, 1),
    (0.6,  0.6, 3, 1),
    (0.3,  0.2, 3, 1),
    (0.3,  0.6, 3, 1),
    (0.2,  0.2, 3, 1),
    (0.2,  0.6, 3, 1),
    (0.04, 0.2, 3, 1),
    (0.04, 0.6, 3, 1),
]

stimuli = Stimuli([20, 40, 60, 80, 100])

ndss = [[NeuroData(os.path.join(data_path.SD_resptostim_const, "{},{},{},{}/[{},0,2,0]".format(*par, s))) for s in [0]+stimuli.S] for par in params]
[[nd.remove_dynamics(500,0) for nd in nds] for nds in ndss]

# consider all 1000 neurons, stimulated or non-stimulated
Cs = [[nd.dynamics.analysis.coherence_parameter(32) for nd in nds] for nds in ndss]
Cs = np.array(Cs)

np.save(os.path.join(data_path.this_dir, "data_CS.npy"), Cs)