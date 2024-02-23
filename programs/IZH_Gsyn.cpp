/**
 * @file IZH_Gsyn.cpp
 * @author likchun@outlook.com
 * @brief numerically simulate the dynamics of a network of spiking neurons
 *        modelled by Izhikevich's model and connected by conductance-based synapses
 *        numerical scheme: weak order 2 Runge-Kutta
 * @version 7
 * @date 2024-Feb-20
 * @note to be compiled in C++ version 11 or later with boost library 1.78.0
 * 
 * Compile command:
 * (MacOS) clang++ ./IZH_Gsyn.cpp -std=c++17 -O3 -o ./IZH_Gsyn.o -I"/Users/likchun/Libraries/c++/boost_1_78_0"
 * 
 */


#include "boost/random.hpp"
#include <fstream>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#elif __APPLE__ || __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif


std::string code_ver = "Version 7\nLast Update: 20 Feburary 2024\n";
std::string prog_info = "This program simulates the dynamics of a network\nof spiking neurons modelled by Izhikevich's model\nand connected by conductance-based synapse model\n";

namespace model_param
{
    namespace Izh
    {
        const double V_s = 30; // spike potentil threshold, mV

        namespace exc {
            const double a = 0.02; // reciprocal of recovery variable decay time constant
            const double b = 0.2; // recovery variable coupling parameter
            const double c = -65.0; // spike-triggered reset potential
            const double d = 8.0; // spike-triggered recovery variable increment
        }
        namespace inh {
            const double a = 0.1; // reciprocal of recovery variable decay time constant
            const double b = 0.2; // recovery variable coupling parameter
            const double c = -65.0; // spike-triggered reset potential
            const double d = 2.0; // spike-triggered recovery variable increment
        }
    }

    namespace Gsynap
    {
        const double V_E = 0; // reversal_potential of the EXC synapses, mV
        const double tau_GE = 5; // conductance decay time constant of the EXC synapses, ms
        const double V_I = -80; // reversal_potential of the INH synapses, mV
        const double tau_GI = 6; // conductance decay time constant of the INH synapses, ms
    }
}


#define _CRT_SECURE_NO_WARNINGS
#define NO_MAIN_ARGUMENT                  argc == 1

/* I-O files & directories */
#define IN_FNAME_PARAMETERS                "vars.txt"
#define OUT_FOLDER                         "output"
#ifdef _WIN32 // For Windows
#define OUT_FNAME_INFO                     "output\\info.txt"
#define OUT_FNAME_SETTINGS                 "output\\sett.json"
#define OUT_FNAME_CONTINUATION             "output\\cont.dat"
#define OUT_FNAME_SPIKE_TIMESTEP           "output\\spks.txt"
#define OUT_FNAME_SPIKE_TIME               "output\\spkt.txt"
#define OUT_FNAME_POTENTIAL_SERIES         "output\\memp.bin"
#define OUT_FNAME_RECOVERY_SERIES          "output\\recv.bin"
#define OUT_FNAME_CURRENT_SYNAP_SERIES     "output\\isyn.bin"
#define OUT_FNAME_CONDUCTANCE_EXC_SERIES   "output\\gcde.bin"
#define OUT_FNAME_CONDUCTANCE_INH_SERIES   "output\\gcdi.bin"
#define OUT_FNAME_CURRENT_STOCH_SERIES     "output\\istc.bin"
#define CREATE_OUTPUT_DIRECTORY(__DIR)    if (CreateDirectoryA(__DIR, NULL) || ERROR_ALREADY_EXISTS == GetLastError()) {}\
                                         else { error_handler::throw_error("dir_create", __DIR); }
#elif __APPLE__ || __linux__ // For Mac & Linux
struct stat st = {0};
#define OUT_FNAME_INFO                     "output/info.txt"
#define OUT_FNAME_SETTINGS                 "output/sett.json"
#define OUT_FNAME_CONTINUATION             "output/cont.dat"
#define OUT_FNAME_SPIKE_TIMESTEP           "output/spks.txt"
#define OUT_FNAME_SPIKE_TIME               "output/spkt.txt"
#define OUT_FNAME_POTENTIAL_SERIES         "output/memp.bin"
#define OUT_FNAME_RECOVERY_SERIES          "output/recv.bin"
#define OUT_FNAME_CURRENT_SYNAP_SERIES     "output/isyn.bin"
#define OUT_FNAME_CONDUCTANCE_EXC_SERIES   "output/gcde.bin"
#define OUT_FNAME_CONDUCTANCE_INH_SERIES   "output/gcdi.bin"
#define OUT_FNAME_CURRENT_STOCH_SERIES     "output/istc.bin"
#define CREATE_OUTPUT_DIRECTORY(__DIR)    if (stat(__DIR, &st) == -1) { mkdir(__DIR, 0700); }
#endif

#define TIMESERIES_BUFFSIZE_THRESH 150000000

namespace datatype_precision
{
    constexpr const int	DIGIT_FLOAT	 = 9;	// use SINGLE floating point precision for time series output
    constexpr const int	DIGIT_DOUBLE = 17;	// use DOUBLE floating point precision for time series output
    const double		PRECISION_FLOAT	 = pow(10, DIGIT_FLOAT);
    const double		PRECISION_DOUBLE = pow(10, DIGIT_DOUBLE);
};

namespace error_handler
{
    void throw_warning(std::string _warn, std::string _e, double _eno)
    {
        std::string msg;
        if (_warn == "param_value") {
            msg = "<invalid parameter value>: \""+_e
                +"\" is given an invalid value of "
                +std::to_string(static_cast<long long>(_eno));
        } else {
            msg = "<unknown>: warnf1";
        }
        std::cerr << "\nWarning" << msg << std::endl;
    }

    void throw_error(std::string _err, std::string _e)
    {
        std::string msg;
        if (_err == "file_access") {
            msg = "<file not found>: \""+_e+"\" cannot be found or accessed";
        } else if (_err == "dir_create") {
            msg = "<io>: directroy \""+_e+"\" cannot be created";
        } else {
            msg = "<unknown failure>: errf1";
        }
        std::cerr << "\nError" << msg << std::endl;
        exit(1);
    }

    template<typename T>
    void throw_error(std::string _err, std::vector<T> &&_e)
    {
        std::string msg;
        if (_err == "neuron_type") {
            msg = "<neuron classification>: inconsistent neuron type is detected at ("
                +std::to_string(static_cast<long long>(_e[0]))+", "
                +std::to_string(static_cast<long long>(_e[1]))+
                ") of the coupling strength matrix";
        } else if (_err == "stimu_input") {
            msg = "<stimulus input>: the length of input stimulus time series ("
                +std::to_string(static_cast<long long>(_e[0]))+") does not match the numerical setting ("
                +std::to_string(static_cast<long long>(_e[1]))+
                ")";
        } else {
            msg = "<unknown failure>: errf2";
        }
        std::cerr << "\nError" << msg << std::endl;
        exit(1);
    }
};

namespace tools
{
    std::string remove_whitespace(std::string &str)
    {
        str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
        return str;
    }

    std::vector<int> string_to_vector_index(std::string &string_of_values, char delimiter)
    {
        std::vector<int> res;
        std::string value;
        std::stringstream ss(string_of_values);
        while (getline(ss, value, delimiter)) {
            res.push_back(std::stoi(tools::remove_whitespace(value))-1); }
        return res;
    }

    template<typename T>
    void print_array(std::vector<T> &array)
    { for (size_t i = 0; i < array.size(); ++i) { std::cout << array[i] << '\t'; } }

    template<typename T>
    void print_array(std::vector<T> &array, size_t print_len)
    { for (size_t i = 0; i < print_len; ++i) { std::cout << array[i] << '\t'; } }

    template<typename T>
    void export_array2D(std::vector<std::vector<T>> &array2D, std::string &&filename, char delimiter)
    {
        std::ofstream file;
        file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file.open(filename, std::ios::trunc);
            file << array2D[0].size();
            for (auto &elem : array2D[0]) { file << delimiter << elem; }
            for (size_t row = 1; row < array2D.size(); ++row) {
                file << '\n' << array2D[row].size();
                for (auto &elem : array2D[row]) { file << delimiter << elem; }
            }
            file.close();
        } catch(std::ofstream::failure const&) {
            error_handler::throw_error("file_access", filename);
        }
    }
};


/* Get all parameters and settings from "vars.txt" */
class Parameters
{
private:

    std::vector<std::string> input_param;

public:

    int network_size = 0;

    const std::string infile_weights;
    const std::string infile_weights_format;
    const char        infile_weights_delim;
    const double      weights_scale_factor_exc;
    const double      weights_scale_factor_inh;

    const double duration;
    const double stepsize;

    const double noise_intensity;
    const double rng_seed;

    const double current_const;

    const std::string infile_stimulus;

    const double init_potential;
    const double init_recovery;

    const double trunc_time_inh;
    const double trunc_time_exc;

    const std::string outfile_info;
    const std::string outfile_sett;
    const std::string outfile_cont;
    const std::string outfile_spike_timestep;
    const std::string outfile_potential_series;
    const std::string outfile_recovery_series;
    const std::string outfile_current_synap_series;
    const std::string outfile_conductance_exc_series;
    const std::string outfile_conductance_inh_series;
    const std::string outfile_current_stoch_series;

    const bool        outPotentialSeries;
    const bool        outRecoverySeries;
    const bool        outCurrentSynapSeries;
    const bool        outConductanceEXCSeries;
    const bool        outConductanceINHSeries;
    const bool        outCurrentStochSeries;

    const std::string program_name;

    Parameters(std::string filename, std::string program_name, bool suppressConsoleMsg=false) :
        input_param(get_input_parameters(filename)),

        infile_weights(input_param[0]),
        infile_weights_format(input_param[1]),
        infile_weights_delim(input_param[2] == "tab" ? '\t' : "space" ? ' ' : input_param[3].c_str()[0]),
        weights_scale_factor_exc(stod(input_param[3])),
        weights_scale_factor_inh(stod(input_param[4])),

        duration(stod(input_param[5])),
        stepsize(stod(input_param[6])),

        noise_intensity(stod(input_param[7])),
        rng_seed(stod(input_param[8])),

        current_const(stod(input_param[9])),

        infile_stimulus(input_param[10]),

        init_potential(stod(input_param[11])),
        init_recovery(stod(input_param[12])),

        trunc_time_inh(stod(input_param[13])),
        trunc_time_exc(stod(input_param[14])),

        outfile_info(OUT_FNAME_INFO),
        outfile_sett(OUT_FNAME_SETTINGS),
        outfile_cont(OUT_FNAME_CONTINUATION),
        outfile_spike_timestep(OUT_FNAME_SPIKE_TIMESTEP),
        outfile_potential_series(OUT_FNAME_POTENTIAL_SERIES),
        outfile_recovery_series(OUT_FNAME_RECOVERY_SERIES),
        outfile_current_synap_series(OUT_FNAME_CURRENT_SYNAP_SERIES),
        outfile_conductance_exc_series(OUT_FNAME_CONDUCTANCE_EXC_SERIES),
        outfile_conductance_inh_series(OUT_FNAME_CONDUCTANCE_INH_SERIES),
        outfile_current_stoch_series(OUT_FNAME_CURRENT_STOCH_SERIES),

        outPotentialSeries((input_param[15] == "true") ? true : false),
        outRecoverySeries((input_param[16] == "true") ? true : false),
        outCurrentSynapSeries((input_param[17] == "true") ? true : false),
        outConductanceEXCSeries((input_param[18] == "true") ? true : false),
        outConductanceINHSeries((input_param[19] == "true") ? true : false),
        outCurrentStochSeries((input_param[20] == "true") ? true : false),

        program_name(program_name)

    { if (!suppressConsoleMsg) { std::cout << "OKAY, parameters imported from \"" << filename << "\"\n"; }}

private:

    std::vector<std::string> get_input_parameters(std::string filename)
    {
        std::vector<std::string> _input_param;
        std::string line, value;
        std::ifstream ifs(filename);
        if (ifs.is_open()) {
            while (std::getline(ifs, line, '\n')) {
                if (line.find('=') != std::string::npos) {
                    std::stringstream ss(line);
                    std::getline(ss, value, '=');
                    std::getline(ss, value, '=');
                    _input_param.push_back(tools::remove_whitespace(value));
            }}
            ifs.close();
        } else {
            error_handler::throw_error("file_access", filename);
        }
        return _input_param;
    }
};


namespace fileio
{
    /* Import all synaptic weights of a network from a text file (.txt).
    There are two formats this program can read:
    1. "nonzero"
        The first line stores the network size / number of neurons.
        Each remaining line stores the a nonzero link:
        {j i w}, where "j" is the outgoing neuron, "i" is the incoming neuron,
        and "w" is the synaptic weight / coupling strength from neuron j to i.
        The delimiter can be specified.
    2. "full"
        The file stores all synaptic weights in N rows and N columns, where N
        is the network size / number of neurons. It's just the basic matrix
        representation we use daily. */
    void import_synaptic_weights(Parameters &par, std::vector<std::vector<double>> &synaptic_weights)
    {
        std::ifstream ifs;
        ifs.open(par.infile_weights, std::ios::in);
        if (ifs.is_open()) {
            std::vector<std::vector<double>> _synaptic_weights;
            std::vector<double> row_buf;
            std::string line, elem;
            if (par.infile_weights_format == "nonzero") {
                std::getline(ifs, line, '\n');
                par.network_size = static_cast<int>(stoi(line));
                synaptic_weights = std::vector<std::vector<double>>(
                    par.network_size, std::vector<double>(par.network_size, 0));
                while(std::getline(ifs, line, '\n')) {
                    std::stringstream ss(line);
                    while(std::getline(ss, elem, par.infile_weights_delim)) {
                        if (elem != "") {
                            row_buf.push_back(std::stod(tools::remove_whitespace(elem)));
                        }
                    }
                    _synaptic_weights.push_back(row_buf);
                    row_buf.clear();
                }
                for (size_t i = 0; i < _synaptic_weights.size(); ++i) {
                    synaptic_weights[static_cast<int>(_synaptic_weights[i][1])-1][static_cast<int>(_synaptic_weights[i][0])-1] = _synaptic_weights[i][2];
                }
            } else if (par.infile_weights_format == "full") {
                while(std::getline(ifs, line, '\n')) {
                    std::stringstream ss(line);
                    while(std::getline(ss, elem, par.infile_weights_delim)) {
                        row_buf.push_back(stod(elem)); }
                    _synaptic_weights.push_back(row_buf);
                    row_buf.clear();
                }
                synaptic_weights = _synaptic_weights;
                par.network_size = synaptic_weights.size();
            }
            ifs.close();
        } else {
            error_handler::throw_error("file_access", par.infile_weights);
        }
    }

    void import_stimulus(Parameters &par, std::vector<int> &stimulated_neuron_idx, std::vector<double> &stimulus_series, std::vector<std::string> &stimulus_info)
    {
        std::ifstream ifs;
        std::string line, datum;
        ifs.open(par.infile_stimulus, std::ios::in);
        if (ifs.is_open()) {
            // get stimulus info
            std::getline(ifs, line, '\n'); // 1st line
            std::stringstream stim_info(line);
            while(std::getline(stim_info, datum, '\t')) {
                if (datum != "") {
                    stimulus_info.push_back(datum);
            }}
            // get indices of stimulated neurons
            std::getline(ifs, line, '\n'); // 2nd line
            std::stringstream stim_nidx(line);
            while(std::getline(stim_nidx, datum, '\t')) {
                if (datum != "") {
                    stimulated_neuron_idx[std::stoi(tools::remove_whitespace(datum))] = 1;
            }}
            // get stimulus time series (assuming identical stimulus is applied to all simulated neurons)
            std::getline(ifs, line, '\n'); // 3rd line
            std::stringstream stim_series(line);
            while(std::getline(stim_series, datum, '\t')) {
                if (datum != "") {
                    stimulus_series.push_back(std::stod(tools::remove_whitespace(datum)));
            }}
            ifs.close();
            if (stimulus_series.size()!=static_cast<int>(par.duration/par.stepsize)) {
                error_handler::throw_error("stimu_input", std::vector<int>({(int)(stimulus_series.size()), (int)(par.duration/par.stepsize)}));
            }
        } else {
            error_handler::throw_error("file_access", par.infile_stimulus);
        }
    }

    void export_file_info(Parameters &par, int continuation, double time_elapsed, int trunc_step_inh, int trunc_step_exc)
    {
        char datetime_buf[64];
        time_t datetime = time(NULL);
        struct tm *tm = localtime(&datetime);
        strftime(datetime_buf, sizeof(datetime_buf), "%c", tm);

        if (continuation == -1) {
            std::ofstream ofs;
            ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            try {
                ofs.open(par.outfile_info, std::ios::trunc);
                ofs << code_ver << '\n'
                    << "------------------------------------------------------------\n"
                    << "program name:\t\t\t" << par.program_name << '\n'
                    << "computation finished at: " << datetime_buf << '\n'
                    << "time elapsed: " << time_elapsed << " s\n\n"
                    << "[network and synaptic weights]" << '\n'
                    << "network file:\t\t\t" << par.infile_weights << '\n'
                    << "number of neurons:\t\t" << par.network_size << '\n'
                    << "exc weight scale factor:\t" << par.weights_scale_factor_exc << "\n"
                    << "inh weight scale factor:\t" << par.weights_scale_factor_inh << "\n\n"
                    << "[numerical settings]" << '\n'
                    << "time step size:\t\t\t" << par.stepsize << " ms" << '\n'
                    << "duration:\t\t\t" << par.duration << " ms" << "\n\n"
                    << "[inputs to neurons]" << '\n'
                    << "random number seed:\t\t" << par.rng_seed << '\n'
                    << "noise intensity:\t\t" << par.noise_intensity << '\n'
                    << "constant current:\t\t" << par.current_const << '\n'
                    << "stimulus file:\t\t\t" << par.infile_stimulus << "\n\n"
                    << "[initial values]\n"
                    << "membrane potential:\t\t" << par.init_potential << " mV" << '\n'
                    << "recovery variable:\t\t" << par.init_recovery << "\n\n"
                    << "[other settings]" << '\n'
                    << "spike truncation time:\t\t" << trunc_step_inh*par.stepsize << " ms" << " (inh)\n"
                    << "spike truncation time:\t\t" << trunc_step_exc*par.stepsize << " ms" << " (exc)\n"
                    << std::endl;
                ofs.close();
            } catch(std::ifstream::failure const&) {
                error_handler::throw_error("file_access", par.outfile_info);
            }
        } else {
            std::ofstream ofs;
            ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            try {
                ofs.open(par.outfile_info, std::ios::app);
                ofs << "------------------------------------------------------------\n"
                    << "computation finished at: " << datetime_buf
                    << "time elapsed: " << time_elapsed << " s\n\n"
                    << "extend duration to:\t\t" << par.duration << " ms\n\n";
                ofs.close();
            } catch(std::ifstream::failure const&) {
                error_handler::throw_error("file_access", par.outfile_info);
            }
        }
    }

    void export_file_json(Parameters &par, int trunc_step_inh, int trunc_step_exc)
    {
        std::ofstream ofs;
        ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs.open(par.outfile_sett, std::ios::trunc);
            ofs << "{"
                << "\"program_name\": " << "\"" << par.program_name << "\"" << ", "
                << "\"network_file\": " << "\"" << par.infile_weights << "\"" << ", "
                << "\"stimulus_file\": " << "\"" << par.infile_stimulus << "\"" << ", "
                << "\"num_neuron\": " << par.network_size << ", "
                << "\"weightscale_factor_exc\": " << par.weights_scale_factor_exc << ", "
                << "\"weightscale_factor_inh\": " << par.weights_scale_factor_inh << ", "
                << "\"stepsize_ms\": " << par.stepsize << ", "
                << "\"duration_ms\": " << par.duration << ", "
                << "\"noise_intensity\": " << par.noise_intensity << ", "
                << "\"rng_seed\": " << par.rng_seed << ", "
                << "\"const_current\": " << par.current_const << ", "
                << "\"exp_trunc_step_inh\": " << trunc_step_inh << ", "
                << "\"exp_trunc_step_exc\": " << trunc_step_exc << ", "
                << "\"init_potential\": " << par.init_potential << ", "
                << "\"init_recovery\": " << par.init_recovery << ", "
                << "\"data_series_export\": "
                    << "{"
                    << "\"potential\": " << (par.outPotentialSeries ? "true" : "false") << ", "
                    << "\"recovery\": " << (par.outRecoverySeries ? "true" : "false") << ", "
                    << "\"current_synap\": " << (par.outCurrentSynapSeries ? "true" : "false") << ", "
                    << "\"conductance_exc\": " << (par.outConductanceEXCSeries ? "true" : "false") << ", "
                    << "\"conductance_inh\": " << (par.outConductanceINHSeries ? "true" : "false") << ", "
                    << "\"current_stoch\": " << (par.outCurrentSynapSeries ? "true" : "false")
                    << "}"
                << "}" << std::endl;
            ofs.close();
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.outfile_info);
        }
    }

    /* Save the numerical data of the last step, which is useful for
    continuing the numerical simulation. This function will be
    implemented in the future. */
    void export_file_continuation(Parameters &par, int continuation, int trunc_step_inh, int trunc_step_exc,
        std::vector<double> &potential_series, std::vector<double> &recovery_series, std::vector<double> &current_synap_series)
    {
        std::ofstream ofs;
        ofs.precision(datatype_precision::DIGIT_DOUBLE);
        ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs.open(par.outfile_cont, std::ios::trunc);
            ofs << ++continuation << '|' << par.program_name << '\n';
            ofs << par.network_size << '|'
                << par.stepsize << '|' << par.duration << '|'
                << par.rng_seed << '|'
                << par.noise_intensity << '|'
                << trunc_step_inh << '|' << trunc_step_exc << '|'
                << par.weights_scale_factor_exc << '|' << par.weights_scale_factor_inh << '|';
            ofs << '|';
            ofs << par.init_potential << '|' << par.init_recovery << '|'
                << model_param::Izh::exc::a << '|' << model_param::Izh::exc::b << '|'
                << model_param::Izh::exc::c << '|' << model_param::Izh::exc::d << '|'
                << model_param::Izh::inh::a << '|' << model_param::Izh::inh::b << '|'
                << model_param::Izh::inh::c << '|' << model_param::Izh::inh::d << '|'
                << model_param::Izh::V_s << '|'
                << model_param::Gsynap::tau_GE << '|' << model_param::Gsynap::tau_GI << '|'
                << model_param::Gsynap::V_E << '|' << model_param::Gsynap::V_I << '|';
            ofs << par.infile_weights << '|'
                << par.infile_weights_format << '|'
                << par.infile_weights_delim << '|'
                << par.infile_stimulus << '|'
                << par.outfile_info << '|'
                << par.outfile_sett << '|'
                << par.outfile_spike_timestep << '|'
                << par.outfile_potential_series << '|'
                << par.outfile_recovery_series << '|'
                << par.outfile_current_synap_series << std::endl;
            ofs << potential_series[0];
            for (int i = 1; i < par.network_size; ++i) {
                ofs << '\t' << potential_series[i]; }
            ofs << '\n' << recovery_series[0];
            for (int i = 1; i < par.network_size; ++i) {
                ofs << '\t' << recovery_series[i]; }
            ofs << '\n' << current_synap_series[0];
            for (int i = 1; i < par.network_size; ++i) {
                ofs << '\t' << current_synap_series[i]; }
            ofs.close();
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.outfile_cont);
        }
    }

    void export_file_spike_data(Parameters &par, std::vector<std::vector<int>> &spike_timesteps, char delimiter='\t')
    {
        std::ofstream ofs;
        ofs.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            ofs.open(par.outfile_spike_timestep, std::ios::trunc);
            ofs << spike_timesteps[0].size();
            for (auto &elem : spike_timesteps[0]) { ofs << delimiter << elem; }
            for (size_t row = 1; row < spike_timesteps.size(); row++) {
                ofs << '\n' << spike_timesteps[row].size();
                for (auto &elem : spike_timesteps[row]) { ofs << delimiter << elem; }
            }
            ofs.close();
        } catch(std::ifstream::failure const&) {
            error_handler::throw_error("file_access", par.outfile_spike_timestep);
        }
    }
}

void display_info(Parameters &par)
{
    std::cout << "---------------------------------------------\n"
              << "|program name:            " << par.program_name << '\n'
              << "|synaptic weights file:   " << par.infile_weights << '\n'
              << "|number of neurons:       " << par.network_size << '\n'
              << "|exc weight scale factor: " << par.weights_scale_factor_exc << '\n'
              << "|inh weight scale factor: " << par.weights_scale_factor_inh << '\n'
              << "|duration:                " << par.duration << " ms" << '\n'
              << "|time step size:          " << par.stepsize << " ms" << '\n'
              << "|stochastic current:\n"
              << "|>noise intensity:        " << par.noise_intensity << '\n'
              << "|>random number seed:     " << par.rng_seed << '\n'
              << "|constant current:\n"
              << "|>amplitude:              " << par.current_const << '\n'
              << "|externally stimulated?   " << ((par.infile_stimulus=="none") ? "no" : "yes") << '\n';
    if (par.infile_stimulus!="none") { std::cout << "|>stimulus file:          " << par.infile_stimulus << '\n'; }
    std::cout << "|spike truncation time:\n"
              << "|>exc neurons:            " << par.trunc_time_exc << " ms" << '\n'
              << "|>inh neurons:            " << par.trunc_time_inh << " ms" << '\n';
    std::cout << "---------------------------------------------" << std::endl;
}

void display_current_datetime(bool suppressConsoleMsg=false)
{
    char datetime_buf[64];
    time_t datetime = time(NULL);
    struct tm *tm = localtime(&datetime);
    if (!suppressConsoleMsg) { strftime(datetime_buf, sizeof(datetime_buf), "%c", tm); std::cout << datetime_buf << '\n'; }
    else { strftime(datetime_buf, sizeof(datetime_buf), "%b %d %H:%M:%S", tm); std::cout << datetime_buf << ' ' << std::flush; }
}

/* Classify each neuron into INHibitory / EXCitatory / UNCLassified,
   and store the result in "neuron_type". The i-th element of
   the std::vector<int> "neuron_type" stores the type of the (i+1)th
   neuron, which can be -1 (INH) / +1 (EXC) / 0 (UNCL). */
void classify_neuron_type(int network_size, std::vector<std::vector<double>> &synaptic_weights, std::vector<int> &neuron_type)
{
    neuron_type = std::vector<int>(network_size);
    for (size_t i = 0; i < synaptic_weights.size(); i++) {
        for (size_t j = 0; j < synaptic_weights.size(); j++) {
            if (synaptic_weights[i][j] > 0) {
                if (neuron_type[j] == -1) {
                    error_handler::throw_error(
                        "neuron_type", std::vector<int>({(int)(i+1), (int)(j+1)})
                    );
                }
                neuron_type[j] = 1;
            }
            else if (synaptic_weights[i][j] < 0) {
                if (neuron_type[j] == 1) {
                    error_handler::throw_error(
                        "neuron_type", std::vector<int>({(int)(i+1), (int)(j+1)})
                    );
                }
                neuron_type[j] = -1;
    }}}
}

void get_indegree(int network_size, std::vector<std::vector<double>> &synaptic_weights, std::vector<double> &indegree)
{
    indegree = std::vector<double>(network_size);
    for (size_t i = 0; i < synaptic_weights.size(); ++i) {
        for (size_t j = 0; j < synaptic_weights[i].size(); ++j) {
            if (synaptic_weights[i][j] != 0) {
                ++indegree[i];
    }}}
}

void scale_synaptic_weights(Parameters &par, std::vector<std::vector<double>> &synaptic_weights)
{
    for (int i = 0; i < par.network_size; i++) {
        for (int j = 0; j < par.network_size; j++) {
            if (synaptic_weights[i][j] > 0) {
                synaptic_weights[i][j] *= par.weights_scale_factor_exc;
            } else if (synaptic_weights[i][j] < 0) {
                synaptic_weights[i][j] *= par.weights_scale_factor_inh;
    }}}
}

/* Create reference "inhibitory(excitatory)_links_index" which stores
   N lists of indices. The i-th list of indices is all the indices of
   inhibitory (excitatory) presynaptic neuron directing into the
   (i+1)th neuron. Useful for speeding up the calculation. */
void create_inlink_reference(std::vector<std::vector<double>> &synaptic_weights,
    std::vector<std::vector<int>> &inhibitory_links_index, std::vector<std::vector<int>> &excitatory_links_index)
{
    std::vector<int> _inh_temp, _exc_temp;
    for (size_t i = 0; i < synaptic_weights.size(); i++) {
        for (size_t j = 0; j < synaptic_weights.size(); j++) {
            if (synaptic_weights[i][j] < 0) { _inh_temp.push_back(j); }
            else if (synaptic_weights[i][j] > 0) { _exc_temp.push_back(j); }
        }
        inhibitory_links_index.push_back(_inh_temp);
        excitatory_links_index.push_back(_exc_temp);
        _inh_temp.clear();
        _exc_temp.clear();
    }
}

/* Estimate the truncation steps needed for the calculation. It uses the
   precision of floating point number / double as a reference to determine
   the threshold of truncation steps that gives accurate results. */
void setup_truncation_step(Parameters &par, std::vector<std::vector<double>> &synaptic_weights,
    int &trunc_step_inh, int &trunc_step_exc)
{
    double w_inh = 0, w_inh_max = 0, w_exc = 0, w_exc_max = 0;
    for (size_t i = 0; i < synaptic_weights.size(); i++) {
        for (size_t j = 0; j < synaptic_weights[i].size(); j++) {
            if (synaptic_weights[i][j] < 0) { w_inh += synaptic_weights[i][j]; }
            else if (synaptic_weights[i][j] > 0) { w_exc += synaptic_weights[i][j]; }
        }
        if (abs(w_inh) > w_inh_max) { w_inh_max = abs(w_inh); }
        if (abs(w_exc) > w_exc_max) { w_exc_max = abs(w_exc); }
    }
    if (par.trunc_time_inh == -1) {
        trunc_step_inh = model_param::Gsynap::tau_GI * log(
            datatype_precision::PRECISION_DOUBLE * par.weights_scale_factor_inh * w_inh_max
        ) / par.stepsize;
        if (w_inh_max == 0 || par.weights_scale_factor_inh == 0) { trunc_step_inh = 0; }
    } else {
        trunc_step_inh = (par.trunc_time_inh / par.stepsize);
    }
    if (par.trunc_time_exc == -1) {
        trunc_step_exc = model_param::Gsynap::tau_GE * log(
            datatype_precision::PRECISION_DOUBLE * par.weights_scale_factor_exc * w_exc_max
        ) / par.stepsize;
        if (w_exc_max == 0 || par.weights_scale_factor_exc == 0) { trunc_step_exc = 0; }
    } else {
        trunc_step_exc = (par.trunc_time_exc / par.stepsize);
    }
}

/* Create a look-up table for the exponential spike decay factors.
   Avoid calculating the expensive and redundant exponential function
   multiple times. */
void setup_exp_lookup_table(Parameters &par, std::vector<double> &spike_exp_inh,
    std::vector<double> &spike_exp_exc, int &trunc_step_inh, int &trunc_step_exc)
{
    spike_exp_inh = std::vector<double>(trunc_step_inh);
    for (int i = 0; i < static_cast<int>(spike_exp_inh.size()); ++i) {
        spike_exp_inh[i] = exp(-i * par.stepsize / model_param::Gsynap::tau_GI);
    }
    spike_exp_exc = std::vector<double>(trunc_step_exc);
    for (int i = 0; i < static_cast<int>(spike_exp_exc.size()); ++i) {
        spike_exp_exc[i] = exp(-i * par.stepsize / model_param::Gsynap::tau_GE);
    }
}


int main(int argc, char **argv)
{
    bool suppressConsoleMsg = false;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-scm") == 0) { suppressConsoleMsg = true; }
    }

    if (!suppressConsoleMsg) { std::cout << code_ver << "...\n" << prog_info << "...\n"; }
    if (!suppressConsoleMsg) { std::cout << "Started at "; }
    display_current_datetime(suppressConsoleMsg);
    clock_t beg = clock();

    int mode = 0;           // 0: overwrite mode | 1: continue mode (not implemented)
    int continuation = -1;  // count the number of times of continuation (not implemented)

    Parameters par(IN_FNAME_PARAMETERS, argv[0], suppressConsoleMsg);

    CREATE_OUTPUT_DIRECTORY(OUT_FOLDER)

    std::vector<int> neuron_type;
    std::vector<double> indegree;
    std::vector<std::vector<double>> synaptic_weights;
    std::vector<std::vector<int>> inh_links_idx, exc_links_idx;
    fileio::import_synaptic_weights(par, synaptic_weights);
    std::vector<double> stimulus_series; // time-dependent stimulus series
    std::vector<int> stimulated_neuron_idx(par.network_size); // i-th element is 1 if the i-th neuron is simulated, otherwise 0
    std::vector<std::string> stimulus_info;
    if (par.infile_stimulus != "none") {
        fileio::import_stimulus(par, stimulated_neuron_idx, stimulus_series, stimulus_info);
    }
    classify_neuron_type(par.network_size, synaptic_weights, neuron_type);
    get_indegree(par.network_size, synaptic_weights, indegree);
    scale_synaptic_weights(par, synaptic_weights);
    create_inlink_reference(synaptic_weights, inh_links_idx, exc_links_idx);

    int	trunc_step_inh, trunc_step_exc;
    std::vector<double> spike_exp_inh, spike_exp_exc;
    setup_truncation_step(par, synaptic_weights, trunc_step_inh, trunc_step_exc);
    setup_exp_lookup_table(par, spike_exp_inh, spike_exp_exc, trunc_step_inh, trunc_step_exc);
    double spike_contribution_sum, potential_prev;

    const double dt = par.stepsize;
    const double sqrt_dt = sqrt(par.stepsize);
    const int    network_size = par.network_size;
    const double noise_intensity = par.noise_intensity;
    const double current_const = par.current_const;
    const bool   outPotentialSeries = par.outPotentialSeries;
    const bool   outRecoverySeries = par.outRecoverySeries;
    const bool   outCurrentSynapSeries = par.outCurrentSynapSeries;
    const bool   outConductanceEXCSeries = par.outConductanceEXCSeries;
    const bool   outConductanceINHSeries = par.outConductanceINHSeries;
    const bool   outCurrentStochSeries = par.outCurrentStochSeries;

    int	now_step = 0, diff_step;
    const int total_step = static_cast<int>(par.duration/par.stepsize);
    std::vector<double> potential(par.network_size);
    std::vector<double> recovery(par.network_size);
    std::vector<double> current_synap(par.network_size);
    fill(potential.begin(), potential.end(), par.init_potential);
    fill(recovery.begin(), recovery.end(), par.init_recovery);
    fill(current_synap.begin(), current_synap.end(), 0.);
    std::vector<double> conductance_inh(par.network_size);
    std::vector<double> conductance_exc(par.network_size);
    std::vector<double> current_stoch(par.network_size);
    std::vector<std::vector<int>> spike_timesteps(par.network_size);
    std::vector<int> justSpiked(par.network_size);

    boost::random::mt19937 random_generator(par.rng_seed);
    boost::random::normal_distribution<double> norm_dist(0, 1);

    const int conductance_step_size = 100;
    const double ddt = dt/conductance_step_size;
    const double sqrt_ddt = sqrt(dt/conductance_step_size);

    std::vector<float> potential_series_buf, recovery_series_buf, current_synap_series_buf,
                      conductance_exc_series_buf, conductance_inh_series_buf, current_stoch_series_buf;
    std::ofstream file_potential_series, file_recovery_series, file_current_synap_series,
        file_conductance_exc_series, file_conductance_inh_series, file_current_stoch_series;
    if (par.outPotentialSeries) {
        file_potential_series.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file_potential_series.open(par.outfile_potential_series, std::ios::trunc | std::ios::binary);
            file_potential_series.close();
            file_potential_series.open(par.outfile_potential_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) { error_handler::throw_error("file_access", par.outfile_potential_series); }
        potential_series_buf.reserve(static_cast<unsigned int>((TIMESERIES_BUFFSIZE_THRESH / par.network_size + 1) * par.network_size));
        for (int i = 0; i < par.network_size; ++i) { potential_series_buf.push_back(potential[i]); }
    }
    if (par.outRecoverySeries) {
        file_recovery_series.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file_recovery_series.open(par.outfile_recovery_series, std::ios::trunc | std::ios::binary);
            file_recovery_series.close();
            file_recovery_series.open(par.outfile_recovery_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) { error_handler::throw_error("file_access", par.outfile_recovery_series); }
        recovery_series_buf.reserve(static_cast<unsigned int>((TIMESERIES_BUFFSIZE_THRESH / par.network_size + 1) * par.network_size));
        for (int i = 0; i < par.network_size; ++i) { recovery_series_buf.push_back(recovery[i]); }
    }
    if (par.outCurrentSynapSeries) {
        file_current_synap_series.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file_current_synap_series.open(par.outfile_current_synap_series, std::ios::trunc | std::ios::binary);
            file_current_synap_series.close();
            file_current_synap_series.open(par.outfile_current_synap_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) { error_handler::throw_error("file_access", par.outfile_current_synap_series); }
        current_synap_series_buf.reserve(static_cast<unsigned int>((TIMESERIES_BUFFSIZE_THRESH / par.network_size + 1) * par.network_size));
        for (int i = 0; i < par.network_size; ++i) { current_synap_series_buf.push_back(current_synap[i]); }
    }
    if (par.outConductanceEXCSeries) {
        file_conductance_exc_series.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file_conductance_exc_series.open(par.outfile_conductance_exc_series, std::ios::trunc | std::ios::binary);
            file_conductance_exc_series.close();
            file_conductance_exc_series.open(par.outfile_conductance_exc_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) { error_handler::throw_error("file_access", par.outfile_conductance_exc_series); }
        conductance_exc_series_buf.reserve(static_cast<unsigned int>((TIMESERIES_BUFFSIZE_THRESH / par.network_size + 1) * par.network_size));
        for (int i = 0; i < par.network_size; ++i) { conductance_exc_series_buf.push_back(0); }
    }
    if (par.outConductanceINHSeries) {
        file_conductance_inh_series.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file_conductance_inh_series.open(par.outfile_conductance_inh_series, std::ios::trunc | std::ios::binary);
            file_conductance_inh_series.close();
            file_conductance_inh_series.open(par.outfile_conductance_inh_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) { error_handler::throw_error("file_access", par.outfile_conductance_inh_series); }
        conductance_inh_series_buf.reserve(static_cast<unsigned int>((TIMESERIES_BUFFSIZE_THRESH / par.network_size + 1) * par.network_size));
        for (int i = 0; i < par.network_size; ++i) { conductance_inh_series_buf.push_back(0); }
    }
    if (par.outCurrentStochSeries) {
        file_current_stoch_series.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file_current_stoch_series.open(par.outfile_current_stoch_series, std::ios::trunc | std::ios::binary);
            file_current_stoch_series.close();
            file_current_stoch_series.open(par.outfile_current_stoch_series, std::ios::app | std::ios::binary);
        } catch(std::ifstream::failure const&) { error_handler::throw_error("file_access", par.outfile_current_stoch_series); }
        current_stoch_series_buf.reserve(static_cast<unsigned int>((TIMESERIES_BUFFSIZE_THRESH / par.network_size + 1) * par.network_size));
        for (int i = 0; i < par.network_size; ++i) { current_stoch_series_buf.push_back(0.0); }
    }

    int total_spike = 0;
    int progress_step[10];
    for (int i = 1; i < 10; ++i) { progress_step[i] = static_cast<int>(total_step*i/10); }

    if (!suppressConsoleMsg) { display_info(par); }
    if (!suppressConsoleMsg) { std::cout << "Running simulation ...\n"; }
    if (par.outPotentialSeries && !suppressConsoleMsg) { std::cout << "Export membrane potential time series? YES\n"; }
    if (par.outRecoverySeries && !suppressConsoleMsg) { std::cout << "Export recovery variable time series? YES\n"; }
    if (par.outCurrentSynapSeries && !suppressConsoleMsg) { std::cout << "Export presynaptic current time series? YES\n"; }
    if (par.outConductanceEXCSeries && !suppressConsoleMsg) { std::cout << "Export presynaptic EXC conductance time series? YES\n"; }
    if (par.outConductanceINHSeries && !suppressConsoleMsg) { std::cout << "Export presynaptic INH conductance time series? YES\n"; }
    if (par.outCurrentStochSeries && !suppressConsoleMsg) { std::cout << "Export stochastic current time series? YES\n"; }

    clock_t beg_sim = clock();
    double current_stimu = 0;
    double a, b, dv1, dv2, du1, du2, potential_0, recovery_0;

    /* Main calculation loop */
    while (now_step < total_step)
    {
        ++now_step;

        if (!suppressConsoleMsg) {
            for (int n = 1; n < 10; ++n) {
                if (now_step == progress_step[n]) {
                    std::cout << "(" << n << "0%) step: " << now_step << "/" << total_step;
                    std::cout << "\n      total spikes: " << total_spike << "\n";
        }}}

        // Handling spike resets
        for (int i = 0; i < network_size; ++i)
        {
            if (potential[i] >= model_param::Izh::V_s) {
                if (neuron_type[i] == -1) {
                    potential[i] = model_param::Izh::inh::c;
                    recovery[i] += model_param::Izh::inh::d;
                } else {
                    potential[i] = model_param::Izh::exc::c;
                    recovery[i] += model_param::Izh::exc::d;
                }
                spike_timesteps[i].push_back(now_step);
                justSpiked[i] = 1;
                total_spike += 1;
            } else { justSpiked[i] = 0; }
        }

        for (int i = 0; i < network_size; ++i)
        {
            /* exact calculation of conductance */
            conductance_inh[i] = 0;
            for (auto &in_inh : inh_links_idx[i]) {
                spike_contribution_sum = 0;
                for (int spk = spike_timesteps[in_inh].size()-1; spk >= 0; --spk) {
                    diff_step = now_step - spike_timesteps[in_inh][spk];
                    if (diff_step < trunc_step_inh) {
                        spike_contribution_sum += spike_exp_inh[diff_step];
                    } else { break; }
                }
                conductance_inh[i] -= synaptic_weights[i][in_inh] * spike_contribution_sum;
            }

            conductance_exc[i] = 0;
            for (auto &in_exc : exc_links_idx[i]) {
                spike_contribution_sum = 0;
                for (int spk = spike_timesteps[in_exc].size()-1; spk >= 0; --spk) {
                    diff_step = now_step - spike_timesteps[in_exc][spk];
                    if (diff_step < trunc_step_exc) {
                        spike_contribution_sum += spike_exp_exc[diff_step];
                    } else { break; }
                }
                conductance_exc[i] += synaptic_weights[i][in_exc] * spike_contribution_sum;
            }

            /* current, potential and recovery */
            current_synap[i] = (conductance_exc[i] * (model_param::Gsynap::V_E - potential[i]) + conductance_inh[i] * (model_param::Gsynap::V_I - potential[i]));
            current_stoch[i] = noise_intensity * norm_dist(random_generator) * sqrt_dt; // Brownian motion
            if (!(stimulus_series.size()==0)) { current_stimu = stimulated_neuron_idx[i]*stimulus_series[now_step]; }
            if (neuron_type[i] == -1) {
                a = model_param::Izh::inh::a;
                b = model_param::Izh::inh::b;
            } else {
                a = model_param::Izh::exc::a;
                b = model_param::Izh::exc::b;
            }

            // Weak Order 2 Runge-Kutta Method //
            dv1 = 0.04*potential[i]*potential[i]+5*potential[i]+140 - recovery[i] + current_synap[i] + current_const + current_stimu;
            du1 = a * (b * potential[i] - recovery[i]);

            potential_0 = potential[i] + dv1 * dt + current_stoch[i];
            recovery_0 = recovery[i] + du1 * dt;
            dv2 = 0.04*potential_0*potential_0+5*potential_0+140 - recovery_0 + current_synap[i] + current_const + current_stimu;
            du2 = a * (b * potential_0 - recovery_0);

            potential[i] += (dv1+dv2)/2 * dt + current_stoch[i];
            recovery[i] += (du1+du2)/2 * dt;
        }

        /* Here, the membrane potential (and other variables) of all neurons
           in a step will be added to a buffer "voltage_time_series_buffer".
           When the buffer is full (i.e., > TIMESERIES_BUFF), the data will
           be dumped into a binary file and the buffer is cleaned. If the
           amount of data is less than "TIMESERIES_BUFF", or after the
           numerical calculation is finished, there will be some "residue"
           data left in the buffer. Those will be treated later. */
        if (outPotentialSeries) // for >TIMESERIES_BUFF
        {
            for (auto &v : potential) { potential_series_buf.push_back(v); }
            // Flush to output file and clear buffer if size exceed "TIMESERIES_BUFF"
            if (potential_series_buf.size() >= TIMESERIES_BUFFSIZE_THRESH) {
                file_potential_series.write(reinterpret_cast<char*>(&potential_series_buf[0]), potential_series_buf.size()*sizeof(float));
                potential_series_buf.clear();
        }}
        if (outRecoverySeries)
        {
            for (auto &u : recovery) { recovery_series_buf.push_back(u); }
            if (recovery_series_buf.size() >= TIMESERIES_BUFFSIZE_THRESH) {
                file_recovery_series.write(reinterpret_cast<char*>(&recovery_series_buf[0]), recovery_series_buf.size()*sizeof(float));
                recovery_series_buf.clear();
        }}
        if (outCurrentSynapSeries)
        {
            for (auto &i : current_synap) { current_synap_series_buf.push_back(i); }
            if (current_synap_series_buf.size() >= TIMESERIES_BUFFSIZE_THRESH) {
                file_current_synap_series.write(reinterpret_cast<char*>(&current_synap_series_buf[0]), current_synap_series_buf.size()*sizeof(float));
                current_synap_series_buf.clear();
        }}
        if (outConductanceEXCSeries)
        {
            for (auto &i : conductance_exc) { conductance_exc_series_buf.push_back(i); }
            if (conductance_exc_series_buf.size() >= TIMESERIES_BUFFSIZE_THRESH) {
                file_conductance_exc_series.write(reinterpret_cast<char*>(&conductance_exc_series_buf[0]), conductance_exc_series_buf.size()*sizeof(float));
                conductance_exc_series_buf.clear();
        }}
        if (outConductanceINHSeries)
        {
            for (auto &i : conductance_inh) { conductance_inh_series_buf.push_back(i); }
            if (conductance_inh_series_buf.size() >= TIMESERIES_BUFFSIZE_THRESH) {
                file_conductance_inh_series.write(reinterpret_cast<char*>(&conductance_inh_series_buf[0]), conductance_inh_series_buf.size()*sizeof(float));
                conductance_inh_series_buf.clear();
        }}
        if (outCurrentStochSeries)
        {
            for (auto &i : current_stoch) { current_stoch_series_buf.push_back(i); }
            if (current_stoch_series_buf.size() >= TIMESERIES_BUFFSIZE_THRESH) {
                file_current_stoch_series.write(reinterpret_cast<char*>(&current_stoch_series_buf[0]), current_stoch_series_buf.size()*sizeof(float));
                current_stoch_series_buf.clear();
        }}
    }

    if (!suppressConsoleMsg) { std::cout << "Completed, time elapsed: " << (double)(clock() - beg_sim)/CLOCKS_PER_SEC << " s\n"; }
    else { std::cout << "| takes " << std::fixed << std::setprecision(0) << (double)(clock() - beg_sim)/CLOCKS_PER_SEC << "s\n"; }
    if (!suppressConsoleMsg) { std::cout << "Total number of spikes: " << total_spike << "\n"; }
    fileio::export_file_spike_data(par, spike_timesteps);
    if (!suppressConsoleMsg) { std::cout << "OKAY, spike data exported\n"; }
    fileio::export_file_info(par, continuation, (double)(clock() - beg)/CLOCKS_PER_SEC, trunc_step_inh, trunc_step_exc);
    if (!suppressConsoleMsg) { std::cout << "OKAY, simulation info exported\n"; }
    fileio::export_file_json(par, trunc_step_inh, trunc_step_exc);
    if (!suppressConsoleMsg) { std::cout << "OKAY, network and simulation settings exported\n"; }
    fileio::export_file_continuation(par, continuation, trunc_step_inh, trunc_step_exc, potential, recovery, current_synap);
    if (!suppressConsoleMsg) { std::cout << "OKAY, continuation file exported" << std::endl; }

    /* The "residue" data aforementioned will be dumped to the binary file here. */
    if (outPotentialSeries) // for <TIMESERIES_BUFF and residue
    {
        file_potential_series.write(
            reinterpret_cast<char*>(&potential_series_buf[0]),
            potential_series_buf.size()*sizeof(float)
        );
        if (file_potential_series.is_open()) { file_potential_series.close(); }
        if (!suppressConsoleMsg) { std::cout << "OKAY, membrane potential time series exported" << std::endl; }
    }
    if (outRecoverySeries)
    {
        file_recovery_series.write(
            reinterpret_cast<char*>(&recovery_series_buf[0]),
            recovery_series_buf.size()*sizeof(float)
        );
        if (file_recovery_series.is_open()) { file_recovery_series.close(); }
        if (!suppressConsoleMsg) { std::cout << "OKAY, recovery variable time series exported" << std::endl; }
    }
    if (outCurrentSynapSeries)
    {
        file_current_synap_series.write(
            reinterpret_cast<char*>(&current_synap_series_buf[0]),
            current_synap_series_buf.size()*sizeof(float)
        );
        if (file_current_synap_series.is_open()) { file_current_synap_series.close(); }
        if (!suppressConsoleMsg) { std::cout << "OKAY, presynaptic current time series exported" << std::endl; }
    }
    if (outConductanceEXCSeries)
    {
        file_conductance_exc_series.write(
            reinterpret_cast<char*>(&conductance_exc_series_buf[0]),
            conductance_exc_series_buf.size()*sizeof(float)
        );
        if (file_conductance_exc_series.is_open()) { file_conductance_exc_series.close(); }
        if (!suppressConsoleMsg) { std::cout << "OKAY, presynaptic EXC conductance time series exported" << std::endl; }
    }
    if (outConductanceINHSeries)
    {
        file_conductance_inh_series.write(
            reinterpret_cast<char*>(&conductance_inh_series_buf[0]),
            conductance_inh_series_buf.size()*sizeof(float)
        );
        if (file_conductance_inh_series.is_open()) { file_conductance_inh_series.close(); }
        if (!suppressConsoleMsg) { std::cout << "OKAY, presynaptic INH conductance time series exported" << std::endl; }
    }
    if (outCurrentStochSeries)
    {
        file_current_stoch_series.write(
            reinterpret_cast<char*>(&current_stoch_series_buf[0]),
            current_stoch_series_buf.size()*sizeof(float)
        );
        if (file_current_stoch_series.is_open()) { file_current_stoch_series.close(); }
        if (!suppressConsoleMsg) { std::cout << "OKAY, stochastic current time series exported" << std::endl; }
    }

    return EXIT_SUCCESS;
}