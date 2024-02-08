Stimuli created
---------------

Stimulus P1

Name: sinwave_nonneg/stim_p1#

Form: S(t) = A * sin(2 * PI * f * t) + A

Description:
A non-negative sinusoidal function with driving frequency f and amplitude A.
A phase shift of -pi/2 is introduced such that at stimulus onset the magnitude of stimulus is zero. It ensures that there is no discontinuous jump at the stimulus onset.

Configurations:
#=a:
>* time step size dt = 0.05 ms
> total duration = 5000 ms
>> stimulated duration = 4000 ms
>> non-stimulated duration before stimulus = 1000 ms
>> non-stimulated duration after stimulus = 0 ms
#=b:
>* time step size dt = 0.02 ms
> total duration = 5000 ms
>> stimulated duration = 4000 ms
>> non-stimulated duration before stimulus = 1000 ms
>> non-stimulated duration after stimulus = 0 ms
(these values are stored in the first line of each file)
#=c:
>* time step size dt = 0.01 ms
> total duration = 5000 ms
>> stimulated duration = 4000 ms
>> non-stimulated duration before stimulus = 1000 ms
>> non-stimulated duration after stimulus = 0 ms
(these values are stored in the first line of each file)

Parameters used:
#=a:
> A = 4
> f = 3, 5, 7
> m = 0.2, 1
#=b:
> A = 4, 12, 24, 36, 48
> f = 3, 5, 7
> m = 0.2, 1
#=c:
> A = 12, 24, 36, 48
> f = 3, 5, 7
> m = 0.2, 1

----------------------------------------------------------------------

Naming convention:
the first letter "p/c" stands for periodic or constant
the following number, e.g. "1", indicates the specific type or form of the stimulus
the next letter, e.g. "a", differentiates the different settings of the same stimulus, e.g., dt, stim_duration, etc.
the value following "A" corresponds to the stimulus amplitude
the value following "f" is the stimulus driving frequency
the value following "m" is the fraction of stimulated neurons


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