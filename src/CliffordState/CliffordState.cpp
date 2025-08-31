#include "CliffordState.h"
#include <algorithm>

void CliffordState::evolve(const QuantumCircuit& qc, const Qubits& qubits) {
  if (qubits.size() != qc.get_num_qubits()) {
    throw std::runtime_error("Provided qubits do not match size of circuit.");
  }

  QuantumCircuit qc_mapped(qc);
  qc_mapped.resize(num_qubits);
  qc_mapped.apply_qubit_map(qubits);

  evolve(qc_mapped);
}

void CliffordState::evolve(const QuantumCircuit& qc) {
  if (!qc.is_clifford()) {
    throw std::runtime_error("Provided circuit is not Clifford.");
  }

  for (auto const &inst : qc.instructions) {
    evolve(inst);
  }
}

void CliffordState::evolve(const Instruction& inst) {
  std::visit(quantumcircuit_utils::overloaded{
      [this](std::shared_ptr<Gate> gate) { 
      std::string name = gate->label();

      if (name == "H") {
      h(gate->qubits[0]);
      } else if (name == "S") {
      s(gate->qubits[0]);
      } else if (name == "Sd") {
      sd(gate->qubits[0]);
      } else if (name == "CX") {
      cx(gate->qubits[0], gate->qubits[1]);
      } else if (name == "X") {
      x(gate->qubits[0]);
      } else if (name == "Y") {
      y(gate->qubits[0]);
      } else if (name == "Z") {
      z(gate->qubits[0]);
      } else if (name == "CY") {
      cy(gate->qubits[0], gate->qubits[1]);
      } else if (name == "CZ") {
        cz(gate->qubits[0], gate->qubits[1]);
      } else if (name == "SWAP") {
        swap(gate->qubits[0], gate->qubits[1]);
      } else {
        throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to CliffordState.evolve.", name));
      }
      },
        [this](const Measurement& m) { 
          measure(m);
        },
        [this](const WeakMeasurement& m) {
          throw std::runtime_error("Cannot perform weak measurements on Clifford states.");
        }
  }, inst);
}

void CliffordState::evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) {
  throw std::runtime_error("Cannot evolve arbitrary gate on Clifford state.");
}

void CliffordState::sd(uint32_t a) {
  s(a);
  s(a);
  s(a);
}

void CliffordState::x(uint32_t a) {
  h(a);
  z(a);
  h(a);
}

void CliffordState::y(uint32_t a) {
  x(a);
  z(a);
}

void CliffordState::z(uint32_t a) {
  s(a);
  s(a);
}

void CliffordState::sqrtx(uint32_t a) {
  h(a);
  s(a);
  h(a);
}
    
void CliffordState::sqrty(uint32_t a) {
  h(a);
  s(a);
  sqrtx(a);
  sd(a);
  h(a);
}

void CliffordState::sqrtz(uint32_t a) {
  s(a);
}

void CliffordState::sqrtxd(uint32_t a) {
  h(a);
  sd(a);
  h(a);
}

void CliffordState::sqrtyd(uint32_t a) {
  h(a);
  sd(a);
  sqrtxd(a);
  s(a);
  h(a);
}

void CliffordState::sqrtzd(uint32_t a) {
  sd(a);
}

void CliffordState::cx(uint32_t a, uint32_t b) {
  h(a);
  cz(b, a);
  h(a);
}

void CliffordState::cy(uint32_t a, uint32_t b) {
  s(a);
  h(a);
  cz(b, a);
  h(a);
  sd(a);
}

void CliffordState::swap(uint32_t a, uint32_t b) {
  cx(a, b);
  cx(b, a);
  cx(a, b);
}

double CliffordState::mzr_expectation() {
  double e = 0.0;

  for (uint32_t i = 0; i < num_qubits; i++) {
    e += mzr_expectation(i);
  }

  return e/num_qubits;
}

double CliffordState::mxr_expectation(uint32_t a) {
  h(a);
  double p = mzr_expectation(a);
  h(a);
  return p;
}

double CliffordState::mxr_expectation() {
  double e = 0.0;

  for (uint32_t i = 0; i < num_qubits; i++) {
    e += mxr_expectation(i);
  }

  return e/num_qubits;
}

double CliffordState::myr_expectation(uint32_t a) {
  s(a);
  h(a);
  double p = mzr_expectation(a);
  h(a);
  sd(a);
  return p;
}

double CliffordState::myr_expectation() {
  double e = 0.0;

  for (uint32_t i = 0; i < num_qubits; i++) {
    e += myr_expectation(i);
  }

  return e/num_qubits;
}

bool CliffordState::mxr(uint32_t a, std::optional<bool> outcome) {
  h(a);
  bool b = mzr(a, outcome);
  h(a);
  return b;
}

bool CliffordState::myr(uint32_t a, std::optional<bool> outcome) {
  s(a);
  h(a);
  bool b = mzr(a, outcome);
  h(a);
  sd(a);
  return b;
}

bool CliffordState::measure(const Measurement& m) {
  if (m.is_basis()) {
    return mzr(m.qubits[0]);
  } else {
    QuantumCircuit qc(m.qubits.size());

    auto args = argsort(m.qubits);
    m.pauli.value().reduce(true, std::make_pair(&qc, args));

    uint32_t q = std::ranges::min(m.qubits);

    evolve(qc, m.qubits);
    bool outcome = mzr(q, m.outcome);
    evolve(qc.adjoint(), m.qubits);

    return outcome;
  }
}

bool CliffordState::weak_measure(const WeakMeasurement& m) {
  throw std::runtime_error("Cannot call a weak measurement on a Clifford state.");
}

std::complex<double> CliffordState::expectation(const PauliString& pauli) const {
  QuantumCircuit qc(num_qubits);
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  uint32_t q = std::ranges::min(qubits);

  pauli.reduce(true, std::make_pair(&qc, qubits));

  auto self = const_cast<CliffordState*>(this);
  self->evolve(qc);
  double exp = self->mzr_expectation(q);
  self->evolve(qc.adjoint());

  return std::complex<double>(exp, 0.0);
}

double CliffordState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  throw std::runtime_error("Not yet implemented!");
}

std::vector<BitAmplitudes> CliffordState::sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot sample bitstrings for Clifford state with n > 15 qubits.");
  }

  std::vector<double> probs = probabilities();
  auto marginal_probs = marginal_probabilities(supports);

  std::discrete_distribution<> dist(probs.begin(), probs.end()); 
  std::minstd_rand rng(randi());

  std::vector<BitAmplitudes> samples;

  for (size_t i = 0; i < num_samples; i++) {
    size_t z = dist(rng);
    BitString bits = BitString::from_bits(z, num_qubits);
    std::vector<double> amplitudes = {probs[z]};
    for (size_t n = 0; n < supports.size(); n++) {
      amplitudes[n-1] = marginal_probs[n][z];
    }

    samples.push_back({bits, amplitudes});
  }

  return samples;
}

std::vector<double> CliffordState::probabilities() const {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot generate probabilities for Clifford state with n > 15 qubits.");
  }

  size_t b = 1u << num_qubits;
  std::vector<double> probs(b);
  for (size_t i = 0; i < b; i++) {
    double p = 1.0;
    for (size_t q = 0; q < num_qubits; q++) {
      if (std::abs(mzr_expectation(q)) < 1e-5) {
        p *= 0.5;
      }
    }
    probs[i] = p;
  }

  return probs;
}

double CliffordState::purity() const {
  return 1.0;
}

std::shared_ptr<QuantumState> CliffordState::partial_trace(const Qubits& qubits) const {
  throw std::runtime_error("Cannot evaluate partial_trace on Clifford states.");
}
