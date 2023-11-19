Naming convention:
the first letter "p/c" stands for periodic or constant
the following number, e.g. "1", indicates the specific type or form of the stimulus
the next letter, e.g. "a", differentiates the different settings of the same stimulus, e.g., dt, stim_duration, etc.
the value following "A" corresponds to the stimulus amplitude
the value following "f" is the stimulus driving frequency
the value following "m" is the fraction of stimulated neurons


stim_p1:
a: time step size dt = 0.05 ms
b: time step size dt = 0.02 ms




File content:
(1st line) information of the stimulus:
> [0] time step size (ms)
> [1] stimulated duration (ms)
> [2] non-stimulated duration before stimulus (ms)
> [3] non-stimulated duration after stimulus (ms)
> [4] stimulus name/type
> [5] driving frequency f_d (Hz)
> [6] stimulus amplitude A
> [7] stimulus bias B
(2nd line) the indices of stimulated neurons
(3rd line) the stimulus time series, the n-th value is the stimulus magnitude at the n-th time step