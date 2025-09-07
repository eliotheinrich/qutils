#include "QuantumCHPState.h"

std::string QuantumCHPState::to_string() const {
  if (print_mode == 0) {
    return tableau.to_string(true);
  } else if (print_mode == 1) {
    return tableau.to_string(false);
  } else if (print_mode == 2) {
    return tableau.to_string_ops(true);
  } else {
    return tableau.to_string_ops(false);
  }
}

void QuantumCHPState::rref() {
  tableau.rref();
}

void QuantumCHPState::xrref() {
  tableau.xrref();
}

void QuantumCHPState::set_print_mode(const std::string& mode) {
  if (mode == "binary_all") {
    print_mode = 0;
  } else if (mode == "binary") {
    print_mode = 1;
  } else if (mode == "paulis_all") {
    print_mode = 2;
  } else if (mode == "paulis") {
    print_mode = 3;
  } else {
    throw std::runtime_error(fmt::format("Invalid print mode: {}", mode));
  }
}

Statevector QuantumCHPState::to_statevector() const {
  return tableau.to_statevector();
}

void QuantumCHPState::h(uint32_t a) {
  tableau.h(a);
}

void QuantumCHPState::s(uint32_t a) {
  tableau.s(a);
}

void QuantumCHPState::sd(uint32_t a) {
  tableau.s(a);
  tableau.s(a);
  tableau.s(a);
}

void QuantumCHPState::cx(uint32_t a, uint32_t b) {
  tableau.cx(a, b);
}

void QuantumCHPState::cy(uint32_t a, uint32_t b) {
  tableau.s(b);
  tableau.h(b);
  tableau.cz(a, b);
  tableau.h(b);
  tableau.sd(b);
}

void QuantumCHPState::cz(uint32_t a, uint32_t b) {
  tableau.h(b);
  tableau.cx(a, b);
  tableau.h(b);
}

PauliString QuantumCHPState::get_stabilizer(size_t i) const {
  return tableau.stabilizers[i];
}

PauliString QuantumCHPState::get_destabilizer(size_t i) const {
  return tableau.destabilizers[i];
}

double QuantumCHPState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  if (support) {
    Tableau restricted = tableau.partial_trace(to_qubits(support_complement(support.value(), num_qubits)));
    return restricted.bitstring_amplitude(bits);
  } else {
    return tableau.bitstring_amplitude(bits);
  }
}

std::vector<double> QuantumCHPState::probabilities() const {
  if (num_qubits > 15) {
    throw std::runtime_error(fmt::format("Cannot evaluate the probabilities() of a {} > 15 qubit state.", num_qubits));
  }

  size_t b = 1u << num_qubits;
  std::vector<double> probs(b);

  for (size_t z = 0; z < b; z++) {
    BitString bits = BitString::from_bits(num_qubits, z);
    probs[z] = expectation(bits);
  }

  return probs;
}

std::vector<PauliString> QuantumCHPState::stabilizers() const {
  return tableau.stabilizers;
}

void QuantumCHPState::random_clifford(const Qubits& qubits) {
  random_clifford_impl(qubits, *this);
}

double QuantumCHPState::mzr_expectation(uint32_t a) const {
  auto [deterministic, _] = tableau.mzr_deterministic(a);
  Tableau tmp = tableau; // TODO do this without copying?
  if (deterministic) {
    auto [outcome, p] = tmp.mzr(a);
    return 2*int(outcome) - 1.0;
  } else {
    return 0.0;
  }
}

MeasurementData QuantumCHPState::mzr(uint32_t a, std::optional<bool> outcome) {
  return tableau.mzr(a, outcome);
}

double QuantumCHPState::sparsity() const {
  return tableau.sparsity();
}

double QuantumCHPState::entanglement(const QubitSupport& support, uint32_t index) {
  auto qubits = to_qubits(support);
  uint32_t system_size = this->num_qubits;
  uint32_t partition_size = qubits.size();

  // Optimization; if partition size is larger than half the system size, 
  // compute the entanglement for the smaller subsystem
  if (partition_size > system_size / 2) {
    std::vector<uint32_t> qubits_complement;
    for (uint32_t q = 0; q < system_size; q++) {
      if (std::find(qubits.begin(), qubits.end(), q) == qubits.end()) {
        qubits_complement.push_back(q);
      }
    }

    return entanglement(qubits_complement, index);
  }

  int rank = tableau.rank(qubits);

  int s = rank - partition_size;

  return static_cast<double>(s);
}

int QuantumCHPState::xrank() const {
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return tableau.xrank(qubits);
}

int QuantumCHPState::partial_xrank(const Qubits& qubits) const {
  return tableau.xrank(qubits);
}

int QuantumCHPState::rank() const {
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return tableau.rank(qubits);
}

int QuantumCHPState::partial_rank(const Qubits& qubits) const {
  return tableau.rank(qubits);
}

void QuantumCHPState::set_x(size_t i, size_t j, bool v) {
  tableau.stabilizers[i].set_x(j, v);
}

void QuantumCHPState::set_z(size_t i, size_t j, bool v) {
  tableau.stabilizers[i].set_z(j, v);
}

#include <glaze/glaze.hpp>

template<>
struct glz::meta<BitString> {
  static constexpr auto value = glz::object(
    "num_bits", &BitString::num_bits,
    "bits", &BitString::bits
  );
};

template<>
struct glz::meta<PauliString> {
  static constexpr auto value = glz::object(
    "num_qubits", &PauliString::num_qubits,
    "phase", &PauliString::phase,
    "bit_string", &PauliString::bit_string
  );
};

template<>
struct glz::meta<Tableau> {
  static constexpr auto value = glz::object(
    "num_qubits", &Tableau::num_qubits,
    "stabilizers", &Tableau::stabilizers,
    "destabilizers", &Tableau::destabilizers
  );
};

template<>
struct glz::meta<QuantumCHPState> {
  static constexpr auto value = glz::object(
    "tableau", &QuantumCHPState::tableau,
    "print_mode", &QuantumCHPState::print_mode
  );
};

DEFINE_SERIALIZATION(QuantumCHPState);
