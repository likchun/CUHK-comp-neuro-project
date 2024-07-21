#include <cmath>
#include <vector>
#include <fstream>
#include "boost/random.hpp"

// compile: clang++ ./simulation-pXY.cpp -std=c++17 -O3 -o ./simulation-pXY.o -I"/.../boost_1_78_0"


// Change the neuron type X and synapse type Y to obtain pEE, pEI, pIE, pII
int neuron_type  = 1;    // 0: EXC, 1: INH
int synapse_type = 1;    // 0: EXC, 1: INH


namespace Izh {
    double a_E   = .02;  // recovery decay time constant reciprocal, exc
    double a_I   = .1;   // recovery decay time constant reciprocal, inh
    double b     = .2;   // potential recovery coupling
    double c     = -65;  // reset potential, mV
    double d_E   = 8;    // recovery jump, exc
    double d_I   = 2;    // recovery jump, inh
    double V_thr = 30;   // cutoff potential, mV
}
namespace Gsyn {
    double tau_E = 5;    // decay time constant, exc, ms
    double tau_I = 6;    // decay time constant, inh, ms
    double V_E   = 0;    // threshold, potential, exc, mV
    double V_I   = -80;  // threshold potential, inh, mV
}


void export_spike_steps(std::string filepath, std::vector<std::vector<int>>& spike_steps, char delimiter='\t')
{
    std::ofstream file_spike_steps;
    file_spike_steps.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file_spike_steps.open(filepath, std::ios::trunc);
    file_spike_steps << spike_steps[0].size();
    for (auto &elem : spike_steps[0]) { file_spike_steps << delimiter << elem; }
    for (size_t row = 1; row < spike_steps.size(); row++) {
        file_spike_steps << '\n' << spike_steps[row].size();
        for (auto &elem : spike_steps[row]) { file_spike_steps << delimiter << elem; }
    }
    file_spike_steps.close();
}


int main(int argc, char **argv) {

    /* Settings and parameters */

    if (argc < 3) {
        std::cout << "accepts two arguments:\n  [1] synaptic weight\n  [2] noise intensity\n";
        return EXIT_FAILURE;
    }
    const double synaptic_weight = std::stod(argv[1]);
    const double noise_intensity = std::stod(argv[2]); // alpha

    const int conductance_bump_steploc = 5000; // at 5 ms = 0.005 s
    const int total_step = 400000; // duration 400 ms = 0.4 s
    const int num_trial = 100000;
    const double stepsize = .001; // ms
    const double rand_seed = 1;


    double a=-1, d=-1, tau=-1, V_R=-1;
    if (neuron_type == 0) {
        a = Izh::a_E;
        d = Izh::d_E;
    } else if (neuron_type == 1) {
        a = Izh::a_I;
        d = Izh::d_I;
    } else { std::exit(EXIT_FAILURE); }
    if (synapse_type == 0) {
        tau = Gsyn::tau_E;
        V_R = Gsyn::V_E;
    } else if (synapse_type == 1) {
        tau = Gsyn::tau_I;
        V_R = Gsyn::V_I;
    } else { std::exit(EXIT_FAILURE); }

    std::string progname = "none";
    if      ((neuron_type == 0) && (synapse_type == 0)) { progname = "pEE"; }
    else if ((neuron_type == 1) && (synapse_type == 0)) { progname = "pIE"; }
    else if ((neuron_type == 0) && (synapse_type == 1)) { progname = "pEI"; }
    else if ((neuron_type == 1) && (synapse_type == 1)) { progname = "pII"; }
    else    { std::exit(EXIT_FAILURE); }
    std::string filepath_spike_steps = progname+std::string("_")+std::string(argv[1])+std::string("_")+std::string(argv[2]);
    std::cout << "finding " << progname << " for g=" << argv[1] << ", alpha="<< argv[2] << " " << std::flush;

    /* *********************** */

    int now_step = 0;
    std::vector<double> potential(num_trial,-70.); // mV
    std::vector<double> recovery(num_trial,-14.);
    std::vector<double> conductance(total_step,0.);
    std::vector<double> current_synap(num_trial,0.);
    std::vector<double> current_stoch(num_trial,0.);
    double potential_prev = 0.;
    const double sqrt_stepsize = std::sqrt(stepsize);
    std::vector<std::vector<int>> spike_steps(num_trial);

    boost::random::mt19937 rand_num_gen(rand_seed);
    boost::random::normal_distribution<double> norm_dist(0,1);


    /* Main loop */

    for (int now_step = conductance_bump_steploc; now_step < total_step; ++now_step) {
        conductance[now_step] = synaptic_weight*std::exp((conductance_bump_steploc-now_step)*stepsize / tau);
    }

    now_step = 0;
    while (now_step < total_step) {
        ++now_step;

        for (int i = 0; i < num_trial; ++i) {
            if (potential[i] >= Izh::V_thr) {
                potential[i] = Izh::c;
                recovery[i] += d;
                spike_steps[i].push_back(now_step);
        }}

        for (int i = 0; i < num_trial; ++i) {
            current_synap[i] = conductance[now_step] * (V_R - potential[i]);
            current_stoch[i] = noise_intensity * norm_dist(rand_num_gen) * sqrt_stepsize;

            double dv1 = 0.04*potential[i]*potential[i]+5*potential[i]+140 - recovery[i] + current_synap[i];
            double du1 = a * (Izh::b * potential[i] - recovery[i]);

            double potential_0 = potential[i] + dv1 * stepsize + current_stoch[i];
            double recovery_0 = recovery[i] + du1 * stepsize;
            double dv2 = 0.04*potential_0*potential_0+5*potential_0+140 - recovery_0 + current_synap[i];
            double du2 = a * (Izh::b * potential_0 - recovery_0);

            potential[i] += (dv1+dv2)/2 * stepsize + current_stoch[i];
            recovery[i] += (du1+du2)/2 * stepsize;
    }}
    export_spike_steps(filepath_spike_steps,spike_steps);

    std::cout << "- OK\n";
    return EXIT_SUCCESS;
}