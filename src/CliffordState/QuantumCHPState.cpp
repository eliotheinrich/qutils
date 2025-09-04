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

void QuantumCHPState::rowsum(uint32_t q1, uint32_t q2) {
  tableau.rowsum(q1, q2);
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

PauliString QuantumCHPState::get_row(size_t i) const {
  return tableau.rows[i];
}

double QuantumCHPState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  Qubits qubits;
  if (support) {
    qubits = to_qubits(support.value());
  } else {
    qubits = Qubits(num_qubits);
    std::iota(qubits.begin(), qubits.end(), 0);
  }
  double p = 1/std::pow(2.0, partial_xrank(qubits));

  bool in_support = true;
  for (size_t q = 0; q < num_qubits; q++) {
    // Need to check that every z-only stabilizer g acts on |z> as g|z> = |z>. 
    const PauliString& row = tableau.rows[q + num_qubits];
    bool has_x = false;
    for (size_t i = 0; i < num_qubits; i++) {
      if (row.get_x(i)) {
        has_x = true;
      }
    }

    if (has_x) {
      continue;
    }

    // row is now z-only. Count the active bits acted on by a Z-operator

    bool positive = true;
    for (size_t i = 0; i < num_qubits; i++) {
      if (row.get_z(i) && bits.get(i)) {
        positive = !positive;
      }
    }

    if (positive != (row.get_r() == 0)) {
      in_support = false;
      break;
    }
  }

  return in_support ? p : 0.0;
}

std::vector<double> QuantumCHPState::probabilities() const {
  double p = 1/std::pow(2.0, xrank());

  size_t b = 1u << num_qubits;
  std::vector<double> probs(b);

  for (size_t z = 0; z < b; z++) {
    BitString bits = BitString::from_bits(num_qubits, z);
    probs[z] = expectation(bits);
  }

  return probs;
}

std::vector<PauliString> QuantumCHPState::stabilizers() const {
  std::vector<PauliString> stabs(tableau.rows.begin() + num_qubits, tableau.rows.end() - 1);
  return stabs;
}

size_t QuantumCHPState::size() const {
  return tableau.rows.size() - 1;
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
  tableau.set_x(i, j, v);
}

void QuantumCHPState::set_z(size_t i, size_t j, bool v) {
  tableau.set_z(i, j, v);
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
    "track_destabilizers", &Tableau::track_destabilizers,
    "rows", &Tableau::rows
  );
};

template<>
struct glz::meta<QuantumCHPState> {
  static constexpr auto value = glz::object(
    "tableau", &QuantumCHPState::tableau,
    "print_mode", &QuantumCHPState::print_mode
  );
};

std::vector<char> QuantumCHPState::serialize() const {
  std::vector<char> bytes;
  auto write_error = glz::write_beve(*this, bytes);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing QuantumCHPState to binary: \n{}", glz::format_error(write_error, bytes)));
  }
  return bytes;
}

void QuantumCHPState::deserialize(const std::vector<char>& bytes) {
  auto parse_error = glz::read_beve(*this, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error reading QuantumCHPState from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}
