"""
Network_Model.py
================
"""

# The NumPy library
import numpy as np

# We use Matplotlib to make graphs
import matplotlib.pyplot as plt
plt.rc('font', family='Charter', size=20)

# and csv to read delimited data files
import csv

# Other miscellaneous imports
import time, math
from tqdm import tqdm


class Network:

    def adjacency_matrix_from_file(self, filename: str) -> None:
        """
        Read an adjacency matrix from a file, which stores
        only nonzero elements in each row, with format:
            j i w_ij, separated by ` ` (whitespace),
        "j" is the pre-synaptic neuron index,
        "i" is the post-synaptic neuron index,
        "w_ji" is the synaptic weight of the link directing
        from j to i. Our neuron index runs from 1 to N.
        """
        content = list(csv.reader(open(filename, 'r', newline=''), delimiter=' '))
        self.size = int(content[0][0])                                      # the first row is the network size/number of neurons
        self.adjacency_matrix = np.zeros((self.size, self.size))
        for x in content[1:]:                                               # the remaining rows are the links with
            #                         "j"          "i"           "wij"      # non-zero synaptic weights
            self.adjacency_matrix[int(x[1])-1][int(x[0])-1] = float(x[2])   # "-1" as our index runs from 1 to N

    def adjacency_matrix_from_numpy_ndarray(self, adjacency_matrix: np.ndarray) -> None:
        """
        Import our adjacency matrix directly from a numpy
        matrix.
        """
        self.adjacency_matrix = adjacency_matrix
        self.size = adjacency_matrix.size

    @property
    def neuron_type(self) -> list:
        """
        Each neuron can be classified into excitatory if the
        synaptic weights of all outgoing links are positive,
        or inhibitory if those are negative.
        """
        return ['exc' if np.all(col >= 0) else ('inh' if np.all(col <= 0) else 'uncl') for col in self.adjacency_matrix.T]

    @property
    def incoming_degree(self) -> np.ndarray:
        """
        Find the incoming degrees, denoted as k_in, of all
        neurons. The i-th element of the returned array is
        the incoming degree of the (i+1)th neuron (since
        our neuron index starts from 1).
        """
        return np.array([len(row[row != 0]) for row in self.adjacency_matrix])

    @property
    def excitatory_incoming_degree(self) -> np.ndarray:
        """
        Find the excitatory incoming degree of all neurons.
        We only consider the incoming links with weight > 0.
        """
        return np.array([len(row[row > 0]) for row in self.adjacency_matrix])

    @property
    def inhibitory_incoming_degree(self) -> np.ndarray:
        """
        Find the excitatory incoming degree of all neurons.
        We only consider the incoming links with weight < 0.
        """
        return np.array([len(row[row < 0]) for row in self.adjacency_matrix])

    @property
    def outgoing_degree(self) -> np.ndarray:
        """
        Find the outgoing degrees, denoted as k_out, of all
        neurons. The i-th element of the returned array is
        the outgoing degree of the (i+1)th neuron (since
        our neuron index starts from 1).
        """
        return np.array([len(row[row != 0]) for row in self.adjacency_matrix.T])


class NetworkModel:

    def __init__(self, network: Network) -> None:
        """
        Conduct numerical simulation for the given
        network using Izhikevich's spiking neuron model.

        Parameter:
        - network: object of class `Network`, contains
                   the network to be simulated
        """
        self.network = network

        # Constants
        ### Refer to the model and notes
        ### Here, the constants a, b, c, d are Nx1 arrays, of which the
        ### i-th element is the corresponding constant of the (i+1)-th
        ### neuron. Of course, we can use if-else conditions to handle
        ### different types of neurons, but using numpy array is faster
        ### and less verbose.
        self.__const_a = np.array([0.1 if _type=='inh' else 0.02 for _type in self.network.neuron_type])
        self.__const_b = 0.2
        self.__const_c = -65.0
        self.__const_d = np.array([2.0 if _type=='inh' else 8.0 for _type in self.network.neuron_type])
        self.__tau_exc = 5.0
        self.__tau_inh = 6.0
        self.__threshold_potential_exc = 0.0     # V_E = 0
        self.__threshold_potential_inh = -80.0   # V_I = -80

        # Model variables
        ### Each variable below is a Nx1 array, of which the i-th
        ### element is the corresponding variable of the (i+1)-th neuron.
        self.membrane_potential = np.full(self.network.size, -70.0)
        self.recovery_variable  = np.full(self.network.size, -14.0)
        self.synaptic_current   = np.zeros(self.network.size)
        ### The time-stamp of each spike is stored in "self.spike_timestamp"
        self.spike_timestamp    = [[] for i in range(self.network.size)]

        # Other settings
        self.external_const_current = np.array([4,4])
        self.__noise_strength = 0.0
        self.__truncation_time = 500
        self.__rng_seed = 0
        np.random.seed(self.__rng_seed)

    @property
    def __gaussian_noise(self) -> np.ndarray:
        # Generate a random Gaussian number with mean=0, sigma=3
        # as we use white noise to stimulate spontaneous activity.
        # A Nx1 array of Gaussian random numbers is returned.
        return np.random.normal(0.0, self.__noise_strength, size=(self.network.size))

    def set_noise_strength(self, alpha) -> None:
        """
        Set the Gaussian noise strength here.
        `alpha` is the standard deviation of
        the Gaussian random numbers. Default
        value is `3.0`.
        """
        self.__noise_strength = alpha

    def run_simulation(self, duration: float, step_size=0.1) -> None:
        """
        Evoke this method to simulate the dynamics of
        the network using our numerical model.

        Parameters:
        - duration:  the simulation duration (unit: ms)
        - step_size: the simulation step size (unit: ms)

        This is the "improved" version of method
        "run_simulation_prototype". We pre-calculated
        all the exponential decay factors, sum over
        only recent pre-synaptic spikes and update
        the neuron conductance all at once.
        """
        print('Simulating dynamics...')
        start_time = time.time()
        dt = step_size

        self.membrane_potential_series = np.zeros((self.network.size, int(duration/step_size)))
        self.recovery_variable_series = np.zeros((self.network.size, int(duration/step_size)))
        self.synaptic_current_series = np.zeros((self.network.size, int(duration/step_size)))

        ### Optimization ###
        # Since the contribution from the pre-synaptic
        # spikes decay exponentially, we can consider only
        # the recent spikes, instead of all spikes. It
        # allows us to sum over the contributions efficiently.
        ## "spike_recent_history" is a NxT matrix, where
        ## N is the network size, T is the maximum number
        ## of steps we consider. Each element can be 0 or 1.
        ## "spike_recent_history[i][t] == 1" means that
        ## neuron i spikes t steps before.
        spike_recent_history = np.zeros((self.network.size, int(self.__truncation_time/dt)))
        # Pre-calculate all the exponential decay factors 
        # for excitatory and inhibitory cases, so that we
        # don't have to calculate the costly exps repeatedly.
        spike_decay_exps_inh = np.array([np.exp(-t*dt/self.__tau_inh) \
            for t in range(0, int(self.__truncation_time/dt))])
        spike_decay_exps_exc = np.array([np.exp(-t*dt/self.__tau_exc) \
            for t in range(0, int(self.__truncation_time/dt))])

        # t is the step count, goes up by 1 in each loop
        for t in tqdm(range(int(duration/dt))):

            # Make a temporary copy so that the other variables
            # take the membrane potential of the previous step
            potential_prev = self.membrane_potential

            # Update membrane potential
            self.membrane_potential += (0.04 * np.square(self.membrane_potential) \
                + 5 * self.membrane_potential + 140 - self.recovery_variable \
                + self.synaptic_current + self.external_const_current) * dt \
                + self.__gaussian_noise * np.sqrt(dt)

            # Update recovery variable
            self.recovery_variable += self.__const_a * (self.__const_b * potential_prev \
                - self.recovery_variable) * dt

            # Reset the cellular variables, if that neuron spikes
            ## The index of the neurons which spiked in this time step
            spiked_neuron_index = np.argwhere(self.membrane_potential >= 30).flatten()
            self.membrane_potential[spiked_neuron_index] = self.__const_c
            self.recovery_variable[spiked_neuron_index] += self.__const_d[spiked_neuron_index]

            # Sum over all pre-synaptic spikes with corresponding decay factor
            ## The following line calculate: sum_over_k{exp(-(t-t[j,k])/tau)}
            ## for all neurons, where t[j,k] is the timestamp
            ## of the k-th spike of the j-th pre-synaptic neuron.
            spike_decay_factor_sum = np.array(
                [np.dot(spike_recent_history[i], spike_decay_exps_inh) if _type=='inh' \
                    else np.dot(spike_recent_history[i], spike_decay_exps_exc) \
                        for i, _type in enumerate(self.network.neuron_type)])

            # Update conductance
            ## ... and update conductance by multiplying the sum
            ## with the corresponding synaptic weight, that is
            ## to calculate: sum_over_j{w_ij * spike_decay_factor_sum}
            conductance_exc = np.matmul(
                self.network.adjacency_matrix.clip(min=0), spike_decay_factor_sum)
            conductance_inh = np.abs(np.matmul(
                self.network.adjacency_matrix.clip(max=0), spike_decay_factor_sum))

            # Update synaptic current
            self.synaptic_current = conductance_exc * (self.__threshold_potential_exc \
                - potential_prev) - conductance_inh * (potential_prev \
                    - self.__threshold_potential_inh)

            # Update the spike history
            ## Shift all columns to the right by one column,
            ## replace the first column by zeroes.
            spike_recent_history = np.roll(spike_recent_history, 1, axis=1)
            spike_recent_history[:,0] = 0
            for i in spiked_neuron_index:
                spike_recent_history[i,0] = 1
                # Store the spike time-stamps
                self.spike_timestamp[i].append(t*dt)

            self.membrane_potential_series[:,t] = self.membrane_potential

        elapsed_time = time.time() - start_time
        print('Completed.')
        print('Time elapsed: {:.1f} s.'.format(elapsed_time))

    def save_spike_timestamps(self, filename='spike_timestamps.txt') -> None:
        """
        Save the spike time-stamps of each neuron in the
        following format:
            each row represents a neuron, the first number N
            is the number of spikes, the following N values
            are the spike times of that neuron.
        """
        with open(filename, 'w') as fp:
            for row in self.spike_timestamp:
                # number of spikes of neuron i
                fp.write('{:d}'.format(len(row)))
                for spike in row:
                    # spike time-stamps follows
                    fp.write('\t{:.3f}'.format(spike))
                fp.write('\n')
        print('Spike time-stamps saved into \"{}\".'.format(filename))



###################
##### Example #####
###################

# 1. Load a network from the adjacency matrix file
net = Network()
net.adjacency_matrix_from_file(filename="network.txt")
## put the network file you want to use next to this python script

# 2. Use numerical method to simulate the dynamics of the network
simulation_duration = 3000. # ms
net_model = NetworkModel(network=net)
net_model.set_noise_strength(alpha=3.) # the strength of noise
net_model.run_simulation(duration=simulation_duration, step_size=.05)
net_model.save_spike_timestamps("spike_timestamps.txt")

plt.show()