#pragma once

#include "QuantumCircuit.h"
#include "EntanglementEntropyState.hpp"
#include "Random.hpp"

#include <map>
#include <bitset>
#include <iostream>
#include <numbers>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "utils.hpp"

#define QS_ATOL 1e-8

class QuantumState;
      
double renyi_entropy(size_t index, const std::vector<double>& samples, double base=std::numbers::e);
double estimate_renyi_entropy(size_t index, const std::vector<double>& samples, double base=std::numbers::e);
double estimate_mutual_renyi_entropy(size_t index, const std::vector<double>& samplesAB, const std::vector<double>& samplesA, const std::vector<double>& samplesB, double base=std::numbers::e);

using PauliAmplitudes = std::pair<PauliString, std::vector<double>>;
using BitAmplitudes = std::pair<BitString, std::vector<double>>;

using PauliMutationFunc = std::function<void(PauliString&)>;
using ProbabilityFunc = std::function<double(double)>;

using MutualMagicAmplitudes = std::vector<std::vector<double>>; // tA, tB, tAB
using MutualMagicData = std::pair<MutualMagicAmplitudes, MutualMagicAmplitudes>; // t2, t4

struct MeasurementResult {
  Eigen::MatrixXcd proj;
  double prob_zero;
  bool outcome;
  
  MeasurementResult(const Eigen::MatrixXcd& proj, double prob_zero, bool outcome)
  : proj(proj), prob_zero(prob_zero), outcome(outcome) {}
};

using Parameter = std::variant<std::string, double, int>;

template <typename T>
T get(const std::map<std::string, Parameter>& params, std::string key, const T& default_value) {
  if (params.contains(key)) {
    return default_value;
  } else {
    return std::get<T>(params.at(key));
  }
}

struct EvolveOpts {
  EvolveOpts() : 
    return_measurement_outcomes(false),
    return_measurement_probabilities(false),
    simplify_circuit(true),
    dag_direction("random")
  {}

  EvolveOpts(const std::map<std::string, Parameter>& params) {
    return_measurement_outcomes = get<int>(params, "return_measurement_outcomes", false);   
    return_measurement_probabilities = get<int>(params, "return_measurement_probabilities", false);   
    simplify_circuit = get<int>(params, "simplify_circuit", true);
    dag_direction = get<std::string>(params, "dag_direction", "random");
  }

  bool return_measurement_outcomes;
  bool return_measurement_probabilities;
  bool simplify_circuit;
  std::string dag_direction;
};

using MeasurementData = std::pair<bool, double>;
using EvolveResult = std::optional<
  std::variant<
    std::vector<MeasurementData>,
    std::vector<bool>,
    std::vector<double>
  >
>;

#define DEFINE_SERIALIZATION(CLASS_NAME)                                                \
std::vector<char> CLASS_NAME::serialize() const {                                       \
    std::vector<char> bytes;                                                            \
    auto write_error = glz::write_beve(*this, bytes);                                   \
    if (write_error) {                                                                  \
        throw std::runtime_error(fmt::format(                                           \
            "Error writing " #CLASS_NAME " to binary: \n{}",                            \
            glz::format_error(write_error, bytes)));                                    \
    }                                                                                   \
    return bytes;                                                                       \
}                                                                                       \
                                                                                        \
void CLASS_NAME::deserialize(const std::vector<char>& bytes) {                          \
    auto parse_error = glz::read_beve(*this, bytes);                                    \
    if (parse_error) {                                                                  \
        throw std::runtime_error(fmt::format(                                           \
            "Error reading " #CLASS_NAME " from binary: \n{}",                          \
            glz::format_error(parse_error, bytes)));                                    \
    }                                                                                   \
}


class QuantumState : public EntanglementEntropyState, public std::enable_shared_from_this<QuantumState> {
  protected:
    uint32_t num_qubits;

    static bool get_dir(const EvolveOpts& opts);
    static EvolveResult process_measurement_results(const std::vector<MeasurementData>& measurements, const EvolveOpts& opts);

	public:
		uint32_t basis;

		QuantumState()=default;
    ~QuantumState()=default;

		QuantumState(uint32_t num_qubits) : EntanglementEntropyState(num_qubits), num_qubits(num_qubits), basis(1u << num_qubits) {}

    uint32_t get_num_qubits() const {
      return num_qubits;
    }

		virtual std::string to_string() const=0;

    void validate_qubits(const Qubits& qubits) const;

    virtual std::complex<double> expectation(const PauliString& pauli) const=0;

    // NOTE: the default implementation relies on computing the full probabilities(). Providing a more efficient implementation from child classes is important.
    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const;

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const=0;
    virtual std::shared_ptr<QuantumState> partial_trace(const QubitSupport& support) const {
      return partial_trace(to_qubits(support));
    }

		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits)=0;
		virtual void evolve(const Eigen::MatrixXcd& gate);
		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t q);

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits);
		virtual void evolve_diagonal(const Eigen::VectorXcd& gate);

    virtual void evolve(const FreeFermionGate& gate);

		virtual std::optional<MeasurementData> evolve(const Instruction& inst);

    virtual EvolveResult evolve(const QuantumCircuit& circuit, EvolveOpts opts=EvolveOpts());
    virtual EvolveResult evolve(const QuantumCircuit& qc, const Qubits& qubits, EvolveOpts opts=EvolveOpts());


    void _evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits);

    template <typename G>
    void evolve_one_qubit_gate(uint32_t q) {
      validate_qubits({q});
      _evolve(G::value, {q});
    }

    #define DEFINE_ONE_QUBIT_GATE(name, struct)             \
    void name(uint32_t q) {                                 \
      evolve_one_qubit_gate<gates::struct>(q);              \
    }

    DEFINE_ONE_QUBIT_GATE(h, H);
    DEFINE_ONE_QUBIT_GATE(x, X);
    DEFINE_ONE_QUBIT_GATE(y, Y);
    DEFINE_ONE_QUBIT_GATE(z, Z);
    DEFINE_ONE_QUBIT_GATE(sqrtX, sqrtX);
    DEFINE_ONE_QUBIT_GATE(sqrtY, sqrtY);
    DEFINE_ONE_QUBIT_GATE(sqrtZ, sqrtZ);
    DEFINE_ONE_QUBIT_GATE(sqrtXd, sqrtXd);
    DEFINE_ONE_QUBIT_GATE(sqrtYd, sqrtYd);
    DEFINE_ONE_QUBIT_GATE(sqrtZd, sqrtZd);
    DEFINE_ONE_QUBIT_GATE(s, sqrtZ);
    DEFINE_ONE_QUBIT_GATE(sd, sqrtZd);
    DEFINE_ONE_QUBIT_GATE(t, T);
    DEFINE_ONE_QUBIT_GATE(td, Td);

    template <typename G>
    void evolve_two_qubit_gate(uint32_t q1, uint32_t q2) { 
      validate_qubits({q1, q2});
      _evolve(G::value, {q1, q2});
    }

    #define DEFINE_TWO_QUBIT_GATE(name, struct)                  \
    void name(uint32_t q1, uint32_t q2) {                        \
      evolve_two_qubit_gate<gates::struct>(q1, q2);              \
    }

    DEFINE_TWO_QUBIT_GATE(cx, CX);
    DEFINE_TWO_QUBIT_GATE(cy, CY);
    DEFINE_TWO_QUBIT_GATE(cz, CZ);
    DEFINE_TWO_QUBIT_GATE(swap, SWAP);

    virtual void random_clifford(const Qubits& qubits);

    static bool check_forced_measure(bool& outcome, double prob_zero);

    virtual MeasurementData measure(const Measurement& m)=0;
    virtual MeasurementData weak_measure(const WeakMeasurement& m)=0;

    // Helper functions
    MeasurementData measure(const Qubits& qubits, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);
    MeasurementData weak_measure(const Qubits& qubits, double beta, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);

    virtual MeasurementData mzr(uint32_t q, std::optional<bool> outcome=std::nullopt) { return measure(Measurement({q}, std::nullopt, outcome)); }
		virtual MeasurementData wmzr(uint32_t q, double beta, std::optional<bool> outcome=std::nullopt) { return weak_measure(WeakMeasurement({q}, beta, std::nullopt, outcome)); }

    virtual std::vector<BitAmplitudes> sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const;

		virtual std::vector<double> probabilities() const=0;
    virtual std::vector<std::vector<double>> partial_probabilities(const std::vector<QubitSupport>& supports) const;
    virtual std::vector<std::vector<double>> marginal_probabilities(const std::vector<QubitSupport>& supports) const;
    virtual double purity() const=0;

    virtual std::vector<char> serialize() const=0;
    virtual void deserialize(const std::vector<char>& bytes)=0;
};

class not_implemented: public std::logic_error {
  public:
      not_implemented() : std::logic_error("Function not yet implemented.") { };
};

class MagicQuantumState : public QuantumState {
	protected:
    bool use_parent;

  public:
    MagicQuantumState()=default;
    ~MagicQuantumState()=default;
    MagicQuantumState(uint32_t num_qubits) : QuantumState(num_qubits), use_parent(false) {}

    using QuantumState::evolve;  
		virtual EvolveResult evolve(const QuantumCircuit& circuit, EvolveOpts opts=EvolveOpts()) override;
    virtual EvolveResult evolve(const QuantumCircuit& qc, const Qubits& qubits, EvolveOpts opts=EvolveOpts()) override;

    virtual std::vector<PauliAmplitudes> sample_paulis_montecarlo(const std::vector<QubitSupport>& qubits, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<PauliAmplitudes> sample_paulis_exhaustive(const std::vector<QubitSupport>& qubits);
    virtual std::vector<PauliAmplitudes> sample_paulis_exact(const std::vector<QubitSupport>& qubits, size_t num_samples, ProbabilityFunc prob);

    virtual std::vector<PauliAmplitudes> sample_paulis(const std::vector<QubitSupport>& qubits, size_t num_samples) {
      throw not_implemented();
    }

    void set_use_parent_implementation(bool use_parent) {
      this->use_parent = use_parent;
    }

    static double calculate_magic_mutual_information_from_samples(const MutualMagicAmplitudes& samples2, const MutualMagicAmplitudes& samples4);
    static double calculate_magic_mutual_information_from_samples(const MutualMagicData& data) { return calculate_magic_mutual_information_from_samples(data.first, data.second); }

    virtual MutualMagicData magic_mutual_information_samples_montecarlo(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual double magic_mutual_information_montecarlo(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual MutualMagicData magic_mutual_information_samples_exact(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples);
    virtual double magic_mutual_information_exact(const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples);
    virtual double magic_mutual_information_exhaustive(const Qubits& qubitsA, const Qubits& qubitsB);

    virtual std::vector<MutualMagicData> bipartite_magic_mutual_information_samples_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<double> bipartite_magic_mutual_information_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt=std::nullopt);
    virtual std::vector<MutualMagicData> bipartite_magic_mutual_information_samples_exact(size_t num_samples);
    virtual std::vector<double> bipartite_magic_mutual_information_exact(size_t num_samples);
    virtual std::vector<double> bipartite_magic_mutual_information_exhaustive();
};

class DensityMatrix;
class Statevector;
class MatrixProductState;
class MatrixProductMixedState;

class DensityMatrix : public MagicQuantumState {
	public:
		Eigen::MatrixXcd data;

    DensityMatrix()=default;

		DensityMatrix(uint32_t num_qubits);

		DensityMatrix(const Statevector& state);

		DensityMatrix(const QuantumCircuit& circuit);

		DensityMatrix(const DensityMatrix& rho);

    DensityMatrix(const MatrixProductState& mps);

    DensityMatrix(const MatrixProductMixedState& mpo);

		DensityMatrix(const Eigen::MatrixXcd& data);

		virtual std::string to_string() const override;

		DensityMatrix partial_trace_density_matrix(const Qubits& traced_qubits) const;
    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;

		virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m) const;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const;
    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const override;

    using QuantumState::evolve;
		virtual void evolve(const Eigen::MatrixXcd& gate) override;
		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override;
		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) override;
		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) override;

    // Convenience
    virtual EvolveResult evolve(const QuantumCircuit& circuit, EvolveOpts opts=EvolveOpts()) override {
      return MagicQuantumState::evolve(circuit, opts);
    }
    virtual EvolveResult evolve(const QuantumCircuit& circuit, const Qubits& qubits, EvolveOpts opts=EvolveOpts()) override {
      return MagicQuantumState::evolve(circuit, qubits, opts);
    }

		double mzr_prob(uint32_t q, bool outcome) const;
		virtual MeasurementData mzr(uint32_t q, std::optional<bool> outcome=std::nullopt) override;
    virtual MeasurementData measure(const Measurement& m) override;
    virtual MeasurementData weak_measure(const WeakMeasurement& m) override;

		Eigen::VectorXd diagonal() const;

		virtual std::vector<double> probabilities() const override;

    virtual double purity() const override {
      return (data*data).trace().real();
    }

    double trace() const {
      return data.trace().real();
    }

		std::map<uint32_t, double> probabilities_map() const;

    struct glaze;
    virtual std::vector<char> serialize() const override;
    virtual void deserialize(const std::vector<char>& bytes) override;
};

class Statevector : public MagicQuantumState {
	public:
		Eigen::VectorXcd data;

    Statevector()=default;

		Statevector(uint32_t num_qubits);

		Statevector(uint32_t num_qubits, uint32_t qregister);

		Statevector(const QuantumCircuit &circuit);

		Statevector(const Statevector& other);

		Statevector(const Eigen::VectorXcd& vec);

		Statevector(const MatrixProductState& state);

		virtual std::string to_string() const override;

		virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m) const;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const;
    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const override;

    using QuantumState::evolve;
		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override;
		virtual void evolve(const Eigen::MatrixXcd& gate) override;
		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;

		virtual void evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) override;
		virtual void evolve_diagonal(const Eigen::VectorXcd& gate) override;

    virtual EvolveResult evolve(const QuantumCircuit& circuit, EvolveOpts opts=EvolveOpts()) override {
      return MagicQuantumState::evolve(circuit, opts);
    }
    virtual EvolveResult evolve(const QuantumCircuit& circuit, const Qubits& qubits, EvolveOpts opts=EvolveOpts()) override {
      return MagicQuantumState::evolve(circuit, qubits, opts);
    }

		double mzr_prob(uint32_t q, bool outcome) const;
		virtual MeasurementData mzr(uint32_t q, std::optional<bool> outcome=std::nullopt) override;
    virtual MeasurementData measure(const Measurement& m) override;
    virtual MeasurementData weak_measure(const WeakMeasurement& m) override;

		double norm() const;
		void normalize();
		void fix_gauge();

		std::map<uint32_t, double> probabilities_map() const;
		virtual std::vector<double> probabilities() const override;
    virtual double purity() const override { 
      return 1.0; 
    }

		std::complex<double> inner(const Statevector& other) const;

		Eigen::VectorXd svd(const Qubits& qubits) const;

    struct glaze;
    virtual std::vector<char> serialize() const override;
    virtual void deserialize(const std::vector<char>& bytes) override;
};

class MatrixProductStateImpl;

class MatrixProductState : public MagicQuantumState {
	private:

	public:
    std::unique_ptr<MatrixProductStateImpl> impl;
    MatrixProductState();
    ~MatrixProductState();

		MatrixProductState(uint32_t num_qubits, uint32_t max_bond_dimension, double sv_threshold=1e-8);
    MatrixProductState(const MatrixProductState& other);
    MatrixProductState(const Statevector& other, uint32_t max_bond_dimension, double sv_threshold=1e-8);
    MatrixProductState(const std::unique_ptr<MatrixProductStateImpl>& impl);
    MatrixProductState& operator=(const MatrixProductState& other);

    void set_left_ortho_lim(uint32_t q);
    void set_right_ortho_lim(uint32_t q);

    static MatrixProductState ising_ground_state(size_t num_qubits, double h, size_t max_bond_dimension=64, double sv_threshold=1e-8, size_t num_sweeps=10);
    static MatrixProductState xxz_ground_state(size_t num_qubits, double delta, size_t max_bond_dimension=64, double sv_threshold=1e-8, size_t num_sweeps=10);
    static MatrixProductState spin_chain_ground_state(size_t num_qubits, const std::vector<double>& Jx, const std::vector<double>& Jy, const std::vector<double>& Jz, size_t max_bond_dimension=64, double sv_threshold=1e-8, size_t num_sweeps=10);

		virtual std::string to_string() const override;

		virtual double entanglement(const QubitSupport& support, uint32_t index) override;
    std::vector<double> singular_values(uint32_t i) const;
    std::pair<std::vector<size_t>, std::vector<std::complex<double>>> tensor(uint32_t q) const;

    std::vector<std::vector<double>> process_bipartite_bit_samples(const std::vector<size_t>& renyi_indices, const std::vector<BitAmplitudes>& samples) const;
    virtual std::vector<BitAmplitudes> sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const override;

    static double calculate_magic_mutual_information_from_samples2(const std::vector<double>& tAB, const std::vector<double>& tA, const std::vector<double>& tB);
    std::vector<double> process_bipartite_pauli_samples(const std::vector<PauliAmplitudes>& samples) const;
    virtual std::vector<PauliAmplitudes> sample_paulis(const std::vector<QubitSupport>& qubits, size_t num_samples) override;

    MatrixProductState partial_trace_mps(const Qubits& qubits) const;
    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;

    virtual std::complex<double> expectation(const PauliString& p) const override;
    std::complex<double> expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const;
    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const override;

    bool is_pure_state() const;
    Eigen::MatrixXcd coefficients_mixed() const;
    Eigen::VectorXcd coefficients_pure() const;
    double trace() const;
    size_t bond_dimension(size_t i) const;

    void orthogonalize(uint32_t q);

    MatrixProductState concatenate(const MatrixProductState& other) const;
    void conjugate();

    std::complex<double> inner(const MatrixProductState& other) const;

		virtual void evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) override;
		virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override;

    virtual EvolveResult evolve(const QuantumCircuit& circuit, EvolveOpts opts=EvolveOpts()) override {
      return MagicQuantumState::evolve(circuit, opts);
    }
    virtual EvolveResult evolve(const QuantumCircuit& circuit, const Qubits& qubits, EvolveOpts opts=EvolveOpts()) override {
      return MagicQuantumState::evolve(circuit, qubits, opts);
    }

		virtual std::vector<double> probabilities() const override;

    virtual double purity() const override;

    virtual MeasurementData measure(const Measurement& m) override;
    virtual MeasurementData weak_measure(const WeakMeasurement& m) override;

    std::vector<double> get_logged_truncerr();

		void print_mps(bool print_data=false) const;

    bool state_valid();
    void set_debug_level(int i);
    void set_orthogonality_level(int i);

    struct glaze;
    virtual std::vector<char> serialize() const override;
    virtual void deserialize(const std::vector<char>& bytes) override;
};

void single_qubit_random_mutation(PauliString& p);

std::vector<QubitSupport> get_bipartite_supports(size_t num_qubits);

std::tuple<Qubits, Qubits, Qubits> get_traced_qubits(
  const Qubits& qubitsA, const Qubits& qubitsB, size_t num_qubits
);

template <typename T>
std::vector<std::vector<double>> extract_amplitudes(const std::vector<T>& samples) {
  size_t num_samples = samples.size();
  if (num_samples == 0) {
    return {};
  }

  size_t num_supports = samples[0].second.size();
  std::vector<std::vector<double>> amplitudes(num_supports, std::vector<double>(num_samples));

  for (size_t j = 0; j < num_samples; j++) {
    auto [p, t] = samples[j];
    if (t.size() != num_supports) {
      throw std::runtime_error("Malformed Amplitudes.");
    }
    for (size_t i = 0; i < num_supports; i++) {
      amplitudes[i][j] = t[i];
    }
  }

  return amplitudes;
}

static std::vector<double> normalize_pauli_samples(const std::vector<double>& p, size_t num_qubits, double purity) {
  std::vector<double> p_(p.size());

  double N = std::pow(2.0, num_qubits) * purity;
  // Normalize
  for (size_t i = 0; i < p.size(); i++) {
    p_[i] = p[i] * p[i] / N;
  }

  return p_;
}

inline void assert_gate_shape(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
  uint32_t h = 1u << qubits.size();
  if ((gate.rows() != h) || gate.cols() != h) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }
}
