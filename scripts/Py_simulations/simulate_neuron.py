import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, collections as mcol, ticker as tck
import random, time, copy
from tqdm import tqdm
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font', family='Charter', size=15)


class NeuronType:
    # excitatory cells (the one you will use)
    regular_spiking = dict(
        name = 'Regular spiking',
        a = 0.02,
        b = 0.2,
        c = -65.0,
        d = 8.0
    )
    excitatory = copy.deepcopy(regular_spiking)
    excitatory['name'] = 'Excitatory'
    # other excitatory
    intrinsically_bursting = dict(
        name = 'Intrinsically bursting',
        a = 0.02,
        b = 0.2,
        c = -55.0,
        d = 4.0
    )
    chattering = dict(
        name = 'Chattering',
        a = 0.02,
        b = 0.2,
        c = -50.0,
        d = 2.0
    )

    # inhibitory cells (the one you will use)
    fast_spiking = dict(
        name = 'Fast spiking',
        a = 0.1,
        b = 0.2,
        c = -65.0,
        d = 2.0
    )
    inhibitory = copy.deepcopy(fast_spiking)
    inhibitory['name'] = 'Inhibitory'
    # other inhibitory
    low_threshold_spiking = dict(
        name = 'Low threshold spiking',
        a = 0.02,
        b = 0.25,
        c = -65.0,
        d = 2.0
    )

class NeuronModel:

    def __init__(self, neuron_type: dict) -> None:
        """
        To simulate neuronal dynamics with Izhikevich's model.

        Parameters:
        - neuron_type: for example:
            regular spiking / fast spiking, you can choose from the class "NeuronType"
        """
        self.__neuron_type = neuron_type['name']
        self.__const_a = neuron_type['a']
        self.__const_b = neuron_type['b']
        self.__const_c = neuron_type['c']
        self.__const_d = neuron_type['d']

        # Initial values
        self.membrane_potential = -70.0
        self.recovery_variable  = -14.0
        self.synaptic_current   = 0.0

        self.spike_timestamp    = []
        self.membrane_potential_series = []
        self.recovery_variable_series = []
        self.synaptic_current_series = []

        self.__dt = 0.05
        self.__noise_strength = 0.0
        self.__rng_seed = 0
        random.seed(self.__rng_seed)
        np.random.seed(self.__rng_seed)

    def set_random_seed(self, seed) -> None:
        self.__rng_seed = seed
        random.seed(self.__rng_seed)
        np.random.seed(self.__rng_seed)

    def set_noise_intensity(self, alpha) -> None:
        """Set the Gaussian noise strength here. `alpha` is the standard
        deviation of the Gaussian random numbers. Default value is `3.0`."""
        self.__noise_strength = alpha 

    @property
    def __gaussian_noise(self) -> float:
        # Generate a random Gaussian number with mean=0, sigma=noise_strength
        # as we use white noise to stimulate spontaneous activity.
        return np.random.normal(0.0, self.__noise_strength)

    def run_simulation(self, duration: float, step_size=0.1) -> None:
        """The main function to do the numerical simulation.
        Update membrane potential, recovery variable in each step.

        Parameters:
        - duration:  the simulation duration (unit: ms)
        - step_size: the simulation step size (unit: ms)
        """
        print('Simulating dynamics...')
        start_time = time.time()
        self.__duration = duration
        self.__dt = step_size

        # t is the step count, goes up by 1 in each loop
        for t in tqdm(range(int(duration/step_size))):

            # Rest membrane potential and recovery
            # variable after the neuron spikes
            if self.membrane_potential >= 30:
                self.membrane_potential = self.__const_c
                self.recovery_variable += self.__const_d
                self.spike_timestamp.append(t*self.__dt)

            # Make a temporary copy so that the other variables
            # take the membrane potential of the previous step
            potential_prev = self.membrane_potential

            # Update membrane potential, equation (1)
            self.membrane_potential += (0.04 * np.square(self.membrane_potential) \
                + 5 * self.membrane_potential + 140 - self.recovery_variable \
                + self.synaptic_current) * self.__dt \
                + self.__gaussian_noise * np.sqrt(self.__dt)

            # Update recovery variable, equation (2)
            self.recovery_variable += self.__const_a * (self.__const_b \
                * potential_prev - self.recovery_variable) * self.__dt

            # Store the membrane potential and current of each step
            self.membrane_potential_series.append(self.membrane_potential)
            self.recovery_variable_series.append(self.recovery_variable)
            self.synaptic_current_series.append(self.synaptic_current)

        elapsed_time = time.time() - start_time
        print('Completed.')
        print('Time elapsed: {:.1f} s.'.format(elapsed_time))

    def run_simulation_with_presynaptic_input(self, presynaptic_neuron: dict, duration: float, step_size=0.1) -> None:
        """
        Augmented version of `run_simulation`. The neuron receives signals from pre-synaptic neurons.

        Parameters:
        - presynaptic_neuron: contain information of the pre-synaptic neuron
        - duration:  the simulation duration (unit: ms)
        - step_size: the simulation step size (unit: ms)
        """
        print('Simulating dynamics...')
        start_time = time.time()
        self.__duration = duration
        self.__dt = step_size

        external_const_current = copy.deepcopy(self.synaptic_current)

        presyn_weight = presynaptic_neuron['weight']
        if presyn_weight < 0: tau = 6.; threshold_potential = -80.
        else: tau = 5.; threshold_potential = 0.
        presyn_spike_train = np.array(presynaptic_neuron['spike_train'])

        trunc_time = 250
        trunc_step = int(trunc_time/step_size)
        conductance_profile = np.array([np.exp(-t*step_size/tau) for t in range(0, trunc_step)])

        # t is the step count, goes up by 1 in each loop
        for t in tqdm(range(int(duration/step_size))):

            # Calculate conductance from pre-synaptic neuron
            try:
                conductance = np.abs(presyn_weight) * np.dot(conductance_profile, \
                    presyn_spike_train[t:t-trunc_step:-1])
            except ValueError:
                conductance = np.abs(presyn_weight) * np.dot(conductance_profile, \
                    np.pad(presyn_spike_train[t:t-trunc_step:-1], (0, \
                        trunc_step-len(presyn_spike_train[t:t-trunc_step:-1])),
                           'constant'))

            # Update synaptic current, equation (3)
            self.synaptic_current = conductance * (threshold_potential - \
                self.membrane_potential) + external_const_current

            # Rest membrane potential and recovery
            # variable after the neuron spikes
            if self.membrane_potential >= 30:
                self.membrane_potential = self.__const_c
                self.recovery_variable += self.__const_d
                self.spike_timestamp.append(t*self.__dt)

            # Make a temporary copy so that the other variables
            # take the membrane potential of the previous step
            potential_prev = self.membrane_potential

            # Update membrane potential, equation (1)
            self.membrane_potential += (0.04 * np.square(self.membrane_potential) \
                + 5 * self.membrane_potential + 140 - self.recovery_variable \
                + self.synaptic_current) * self.__dt \
                + self.__gaussian_noise * np.sqrt(self.__dt)

            # Update recovery variable, equation (2)
            self.recovery_variable += self.__const_a * (self.__const_b \
                * potential_prev - self.recovery_variable) * self.__dt

            # Store the membrane potential of each step
            self.membrane_potential_series.append(self.membrane_potential)
            self.recovery_variable_series.append(self.recovery_variable)
            self.synaptic_current_series.append(self.synaptic_current)

        elapsed_time = time.time() - start_time
        print('Completed.')
        print('Time elapsed: {:.1f} s.'.format(elapsed_time))

    def save_spike_timestamps(self, filename='spike_timestamps.txt') -> None:
        """Save the spike time-stamps in the following format:
        - the first number N is the number of spikes, the following N values are the spike times."""
        with open(filename, 'w') as fp:
            fp.write('{:d}'.format(len(self.spike_timestamp)))
            for spike in self.spike_timestamp: fp.write('\t{:.3f}'.format(spike))
        print('Spike time-stamps saved into \"{}\".'.format(filename))

    def save_membrane_potential_time_series(self, filename='voltage_series.pkl') -> None:
        """Save membrane potential/voltage time series into a binary file."""
        np.save(self.membrane_potential_time_series, open(filename, 'wb'))

    def draw_time_series(self, filename='time series of variables') -> None:
        """Draw a membrane potential & synaptic current time series,
        which shows how the variables changes over time."""
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(np.array(range(len(self.membrane_potential_series)))*self.__dt,
                self.membrane_potential_series, c='k', label='Membrane potential')
        ax.plot(np.array(range(len(self.recovery_variable_series)))*self.__dt,
                self.recovery_variable_series, c='g', label='Recovery variable')
        ax.plot(np.array(range(len(self.synaptic_current_series)))*self.__dt,
                self.synaptic_current_series, c='darkorange', label='Synaptic current')
        ax.set(xlabel='Time (ms)', ylabel='Variable', xlim=(0, self.__duration))
        ax.set_title('Neuron type: {}'.format(self.__neuron_type), y=1.02, loc='left', fontsize=15)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(filename)

    def draw_raster_plot(self, filename='raster plot') -> None:
        """Draw a raster plot, in which each vertical line
        represents a spike at its corresponding time."""
        fig, ax = plt.subplots(figsize=(10,2))
        ax.eventplot(self.spike_timestamp, colors='k')
        ax.set(xlabel='Time (ms)', ylabel='', xlim=(0, self.__duration))
        ax.set_title('Neuron type: {}'.format(self.__neuron_type), y=1.02, loc='left', fontsize=15)
        ax.set_yticklabels([])
        plt.tight_layout()
        fig.savefig(filename)

    def draw_phase_portrait_2d(self, filename="phase portrait 2d"):
        fig, ax = plt.subplots(figsize=(7,6))
        lc = _colorline(self.membrane_potential_series, self.recovery_variable_series, cmap="copper_r", linewidth=2, ax=ax)
        cbar = plt.colorbar(lc)
        cbar.ax.yaxis.set_major_locator(tck.FixedLocator(cbar.ax.get_yticks().tolist()))
        cbar.ax.set_yticklabels(['{:f}'.format(Decimal('{}'.format(x)).normalize()) for x in np.arange(0, self.__duration+(self.__duration-0)/5, (self.__duration-0)/5)])
        cbar.set_label('Time (ms)', rotation=90, labelpad=15)
        u = np.array([np.amin(self.recovery_variable_series), np.amax(self.recovery_variable_series)])
        u_nullcline = np.vstack((u/self.__const_b, u))
        ax.plot(*u_nullcline, "--", c="navy", label="u nullcline")
        ax.set(xlabel="Membrane potential v (mV)", ylabel="Recovery variable u")
        ax.set_title('Neuron type: {}'.format(self.__neuron_type), y=1.02, loc='left', fontsize=15)
        ax.legend()
        ax.grid(True)
        ax.autoscale()
        plt.tight_layout()
        fig.savefig(filename)

    def draw_phase_portrait_3d(self, duration_ms, timestep_ms, filename="phase portrait 3d"):
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111,projection="3d")
        ax.set_proj_type("ortho")
        ax.plot(self.membrane_potential_series, self.recovery_variable_series, zs=np.arange(0,duration_ms,step=timestep_ms),zdir="z",label="phase plane trajectory",c="k")
        ax.set_title('Neuron type: {}'.format(self.__neuron_type), y=1.02, loc='left', fontsize=15)
        ax.set(xlabel="membrane potential (mV)",ylabel="recovery variable",zlabel="time (ms)")
        ax.legend()
        fig.savefig(filename)

    def reset(self) -> None:
        """Reset the system, with current, noise strength and random number seed unchanged."""
        self.membrane_potential = -70.0
        self.recovery_variable  = -14.0
        self.spike_timestamp    = []
        self.membrane_potential_series = []

def generate_spike_train(duration: float, step_size: float, spike_timestamps: list):
    """
    Return a spike train from a given `spike_timestamps`.
    """
    return [1 if t*step_size in spike_timestamps else 0 for t in range(int(duration/step_size))]

def _colorline(x, y, z=None, cmap="copper", norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1, ax=None):
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = _make_segments(x, y)
    lc = mcol.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    # if ax==None: ax = plt.gca()
    ax.add_collection(lc)
    return lc

def _make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments



"""Run numerical simulation"""
# neuron_model = NeuronModel(NeuronType.inhibitory)
# neuron_model.membrane_potential = -70.
# neuron_model.recovery_variable = -14.
# neuron_model.synaptic_current = 4.
# neuron_model.set_noise_intensity(alpha=0)

# neuron_model.run_simulation(duration=1000., step_size=.005)
# print("Mean firing rate: {} Hz".format(len(neuron_model.spike_timestamp)/10))

# print("Spike time-stamps: {}".format(neuron_model.spike_timestamp))
# neuron_model.save_spike_timestamps('spike_timestamps.txt')
# neuron_model.draw_time_series()
# # neuron_model.draw_raster_plot()
# neuron_model.draw_phase_portrait()
# plt.show()



###################
##### Example #####
###################

"""Run simulation with pre-synaptic current input"""
presynaptic_neuron = {
    # "weight": +0.435,
    # "weight": +0.15,
    "weight": +0,
    "spike_train": generate_spike_train(60., .005, [8.52])
}
neuron_model = NeuronModel(NeuronType.regular_spiking)
neuron_model.membrane_potential = -70.
neuron_model.recovery_variable = -14.
neuron_model.synaptic_current = 20.
neuron_model.set_random_seed(2)
# neuron_model.set_noise_intensity(alpha=6.)
neuron_model.set_noise_intensity(alpha=0.)

neuron_model.run_simulation_with_presynaptic_input(presynaptic_neuron, duration=120., step_size=.005)

print("Spike time-stamps: {}".format(neuron_model.spike_timestamp))
# neuron_model.save_spike_timestamps("spike_timestamps.txt")
neuron_model.draw_time_series()
# neuron_model.draw_raster_plot()
# neuron_model.draw_phase_portrait_2d()
neuron_model.draw_phase_portrait_3d(120.,.005)
plt.show()
