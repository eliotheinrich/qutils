#include "QuantumStates.h"

bool QuantumState::get_dir(const EvolveOpts& opts) {
  std::string direction = opts.dag_direction;

  bool dir;
  if (direction == "random") {
    return randf() < 0.5;
  } else if (direction == "left") {
    return true;
  } else {
    return false;
  }
}

EvolveResult QuantumState::process_measurement_results(const std::vector<MeasurementData>& measurements, const EvolveOpts& opts) {
  if (opts.return_measurement_outcomes && opts.return_measurement_probabilities) {
    return measurements;
  } else if (opts.return_measurement_outcomes) {
    std::vector<bool> outcomes(measurements.size());
    size_t i = 0;
    for (const auto& [outcome, prob] : measurements) {
      outcomes[i++] = outcome;
    }
    return outcomes;
  } else if (opts.return_measurement_probabilities) {
    std::vector<double> probabilities(measurements.size());
    size_t i = 0; 
    for (const auto& [outcome, prob] : measurements) {
      probabilities[i++] = prob;
    }
    return probabilities;
  } else {
    return std::nullopt;
  }
}

std::vector<std::vector<double>> QuantumState::marginal_probabilities(const std::vector<QubitSupport>& supports) const {
  size_t num_supports = supports.size();

  auto partials = partial_probabilities(supports);
  std::vector<std::vector<double>> marginals(num_supports + 1, std::vector<double>(basis, 0.0));
  marginals[0] = partials[0];

  for (size_t i = 0; i < num_supports; i++) {
    auto qubits = to_qubits(supports[i]);
    std::sort(qubits.begin(), qubits.end());

    for (uint32_t z = 0; z < basis; z++) {
      uint32_t zA = quantumstate_utils::reduce_bits(z, qubits);
      marginals[i + 1][z] = partials[i + 1][zA];
    }
  }

  return marginals;
}

std::vector<std::vector<double>> QuantumState::partial_probabilities(const std::vector<QubitSupport>& supports) const {
  std::vector<double> probs = probabilities();

  size_t num_supports = supports.size();

  std::vector<std::vector<double>> partials(num_supports + 1);
  partials[0] = probs;

  for (size_t i = 0; i < num_supports; i++) {
    auto qubits = to_qubits(supports[i]);
    std::sort(qubits.begin(), qubits.end());

    std::vector<double> partial(1u << qubits.size(), 0.0);
    for (uint32_t z = 0; z < basis; z++) {
      uint32_t zA = quantumstate_utils::reduce_bits(z, qubits);
      partial[zA] += probs[z];
    }

    partials[i + 1] = partial;
  }

  return partials;
}

double QuantumState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  auto probs = probabilities();
  return probs[bits.to_integer()];
}

std::vector<BitAmplitudes> QuantumState::sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const {
  auto marginals = marginal_probabilities(supports);
  auto probs = marginals[0];

  std::minstd_rand rng(randi());
  std::discrete_distribution<> dist(probs.begin(), probs.end());

  std::vector<BitAmplitudes> samples;

  for (size_t i = 0; i < num_samples; i++) {
    uint32_t z = dist(rng);
    BitString bits = BitString::from_bits(num_qubits, z);

    std::vector<double> amplitudes = {probs[z]};
    for (size_t j = 1; j < marginals.size(); j++) {
      amplitudes.push_back(marginals[j][z]);
    }

    samples.push_back({bits, amplitudes});
  }

  return samples;
}

void single_qubit_random_mutation(PauliString& p) {
  size_t j = randi() % p.num_qubits;
  size_t g = randi() % 4;

  if (g == 0) {
    p.set_x(j, 0);
    p.set_z(j, 0);
  } else if (g == 1) {
    p.set_x(j, 1);
    p.set_z(j, 0);
  } else if (g == 2) {
    p.set_x(j, 0);
    p.set_z(j, 1);
  } else {
    p.set_x(j, 1);
    p.set_z(j, 1);
  }
}

void QuantumState::validate_qubits(const Qubits& qubits) const {
  for (const auto q : qubits) {
    if (q > num_qubits) {
      throw std::runtime_error(fmt::format("Instruction called on qubit {}, which is not valid for a state with {} qubits.", q, num_qubits));
    }
  }
}

static inline double log(double x, double base) {
  return std::log(x) / std::log(base);
}

double renyi_entropy(size_t index, const std::vector<double>& probs, double base) {
  double s = 0.0;
  if (index == 1) {
    for (auto p : probs) {
      if (p > 1e-14) {
        s += p * log(p, base);
      }
    }
    return -s;
  } else {
    for (auto p : probs) {
      s += std::pow(p, index);
    }
    return log(s, base)/(1.0 - index);
  }
}

double estimate_renyi_entropy(size_t index, const std::vector<double>& samples, double base) {
  if (index == 1) {
    double q = 0.0;
    for (auto p : samples) {
      q += log(p, base);
    }

    q = q/samples.size();
    return -q;
  } else {
    double q = 0.0;
    for (auto p : samples) {
      q += std::pow(p, index - 1.0);
    }

    q = q/samples.size();
    return 1.0/(1.0 - index) * log(q, base);
  }
}

double estimate_mutual_renyi_entropy(size_t index, const std::vector<double>& samplesAB, const std::vector<double>& samplesA, const std::vector<double>& samplesB, double base) {
  size_t N = samplesAB.size();

  if (index == 1) {
    double p = 0.0;
    for (size_t i = 0; i < N; i++) {
      p += log(samplesAB[i] / (samplesA[i] * samplesB[i]), base);
    }
    return p/N;
  } else {
    return estimate_renyi_entropy(index, samplesA, 2) + estimate_renyi_entropy(index, samplesB, 2) - estimate_renyi_entropy(index, samplesAB, 2);
  }
}

void QuantumState::_evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
  size_t r = gate.rows();
  size_t c = gate.cols();
  if (r == c) {
    evolve(gate, qubits);
  } else if (c == 1) {
    evolve_diagonal(gate, qubits);
  } else {
    throw std::runtime_error(fmt::format("Invalid gate shape: {}x{}", r, c));
  }
}

void QuantumState::random_clifford(const Qubits& qubits) {
  Qubits qubits_ = argsort(qubits);
  QuantumCircuit qc(qubits.size());
  random_clifford_impl(qubits_, qc);
  evolve(qc, qubits);
}

void QuantumState::evolve(const Eigen::MatrixXcd& gate) {
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  _evolve(gate, qubits);
}

void QuantumState::evolve(const Eigen::Matrix2cd& gate, uint32_t q) {
  Qubits qubit{q};
  _evolve(gate, qubit); 
}

void QuantumState::evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) { 
  evolve(Eigen::MatrixXcd(gate.asDiagonal()), qubits); 
}

void QuantumState::evolve_diagonal(const Eigen::VectorXcd& gate) { 
  evolve(Eigen::MatrixXcd(gate.asDiagonal())); 
}

void QuantumState::evolve(const FreeFermionGate& gate) {
  evolve(gate.to_gate());
}

std::optional<MeasurementData> QuantumState::evolve(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded{
    [this](std::shared_ptr<Gate> gate) -> std::optional<MeasurementData> { 
      _evolve(gate->define(), gate->qubits); 
      return std::nullopt;
    },
    [this](const FreeFermionGate& gate) -> std::optional<MeasurementData> {
      evolve(gate);
      return std::nullopt;
    },
    [this](const Measurement& m) -> std::optional<MeasurementData> { 
      return measure(m);
    },
    [this](const WeakMeasurement& m) -> std::optional<MeasurementData> {
      return weak_measure(m);
    }
  }, inst);
}

EvolveResult QuantumState::evolve(const QuantumCircuit& circuit, EvolveOpts opts) {
  if (circuit.num_params() > 0) {
    throw std::invalid_argument("Unbound QuantumCircuit parameters; cannot evolve Statevector.");
  }

  BitString bits(circuit.get_num_cbits());

  // Preparing reversed measurement_map
  size_t num_measurements = circuit.get_num_measurements();
  std::vector<MeasurementData> measurements(num_measurements);
  std::map<size_t, size_t> reversed_map;
  std::vector<size_t> measurement_map = circuit.get_measurement_map();
  for (size_t i = 0; i < num_measurements; i++) {
    reversed_map[measurement_map[i]] = i;
  }

  for (size_t i = 0; i < circuit.length(); i++) {
    const auto& cinst = circuit.instructions[i];
    if (!cinst.should_execute(bits)) {
      continue;
    }

    auto result = evolve(cinst.inst);
    if (result) {
      measurements[reversed_map.at(i)] = result.value();
      if (cinst.target) {
        bits.set(cinst.target.value(), result->first);
      }
    }
  }  

  return process_measurement_results(measurements, opts);
}

EvolveResult QuantumState::evolve(const QuantumCircuit& circuit, const Qubits& qubits, EvolveOpts opts) {
  if (qubits.size() != circuit.get_num_qubits()) {
    throw std::runtime_error("Provided qubits do not match size of circuit.");
  }

  QuantumCircuit circuit_mapped(circuit);
  circuit_mapped.resize_qubits(num_qubits);
  circuit_mapped.apply_qubit_map(qubits);

  return evolve(circuit_mapped, opts);
}

bool QuantumState::check_forced_measure(bool& outcome, double prob_zero) {
  if (((1.0 - prob_zero) < QS_ATOL && outcome) || (prob_zero < QS_ATOL && !outcome)) {
    outcome = !outcome;
    throw std::runtime_error("Invalid forced measurement.\n");
    return true;
  }

  return false;
}

MeasurementData QuantumState::measure(const Qubits& qubits, std::optional<PauliString> pauli, std::optional<bool> outcome) {
  return measure(Measurement(qubits, pauli, outcome));
}

MeasurementData QuantumState::weak_measure(const Qubits& qubits, double beta, std::optional<PauliString> pauli, std::optional<bool> outcome) {
  return weak_measure(WeakMeasurement(qubits, beta, pauli, outcome));
}
