#include "Tableau.h"
#include <stdexcept>

void TableauBase::sd(uint32_t a) {
  s(a);
  s(a);
  s(a);
}

void TableauBase::x(uint32_t a) {
  h(a);
  s(a);
  s(a);
  h(a);
}

void TableauBase::y(uint32_t a) {
  h(a);
  s(a);
  s(a);
  h(a);
  s(a);
  s(a);
}

void TableauBase::z(uint32_t a) {
  s(a);
  s(a);
}

void TableauBase::cz(uint32_t a, uint32_t b) {
  h(b);
  cx(a, b);
  h(b);
}

PauliString TableauBase::get_stabilizer(size_t i) const {
  std::vector<Pauli> paulis(num_qubits);
  for (size_t j = 0; j < num_qubits; j++) {
    paulis[j] = get_pauli(i, j);
  }
  uint8_t phase = get_phase(i);
  return PauliString(paulis, phase);
}

Eigen::MatrixXi TableauBase::to_matrix() const {
  Eigen::MatrixXi M = Eigen::MatrixXi::Zero(num_qubits, 2*num_qubits);

  for (size_t i = 0; i < num_qubits; i++) {
    for (size_t j = 0; j < num_qubits; j++) {
      Pauli p = get_pauli(i, j);
      if (p == Pauli::I) {

      } else if (p == Pauli::X) {
        M(i, j + num_qubits) = 1;
      } else if (p == Pauli::Z) {
        M(i, j) = 1;
      } else {
        M(i, j) = 1;
        M(i, j + num_qubits) = 1;
      }
    }
  }

  return M;
}

bool TableauBase::operator==(const TableauBase& other) const {
  for (size_t i = 0; i < num_qubits; i++) {
    if (get_stabilizer(i) != other.get_stabilizer(i)) {
      return false;
    }
  }

  for (size_t i = 0; i < num_qubits; i++) {
    if (get_destabilizer(i) != other.get_destabilizer(i)) {
      return false;
    }
  }
  return true;
}

Statevector TableauBase::to_statevector() const {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot create a Statevector with more than 31 qubits.");
  }

  Eigen::MatrixXcd dm = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);
  Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);

  for (size_t i = 0; i < num_qubits; i++) {
    PauliString p = get_stabilizer(i);
    Eigen::MatrixXcd g = p.to_matrix();
    dm = dm*((I + g)/2.0);
  }

  uint32_t N = 1u << num_qubits;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(dm);
  Eigen::VectorXcd vec = solver.eigenvectors().block(0,N-1,N,1).rowwise().reverse();

  return Statevector(vec);
}

double TableauBase::sparsity() const {
  float nonzero = 0;
  for (uint32_t i = 0; i < num_qubits; i++) {
    for (uint32_t j = 0; j < num_qubits; j++) {
      Pauli p = get_pauli(i, j);
      if (p == Pauli::X || p == Pauli::Z) {
        nonzero += 1;
      } else if (p == Pauli::Y) {
        nonzero += 2;
      }
    }
  }

  return nonzero/(2*num_qubits*num_qubits);
}

Tableau::Tableau(uint32_t num_qubits) : TableauBase(num_qubits) {
  stabilizers = std::vector<PauliString>(num_qubits, PauliString(num_qubits));
  destabilizers = std::vector<PauliString>(num_qubits, PauliString(num_qubits));
  for (uint32_t i = 0; i < num_qubits; i++) {
    destabilizers[i].set_x(i, true);
    stabilizers[i].set_z(i, true);
  }
}

Pauli Tableau::get_pauli(size_t i, size_t j) const {
  return stabilizers[i].to_pauli(j);
}

PauliString Tableau::get_stabilizer(size_t i) const {
  return stabilizers[i];
}

PauliString Tableau::get_destabilizer(size_t i) const {
  return destabilizers[i];
}

uint8_t Tableau::get_phase(size_t i) const {
  return stabilizers[i].get_r();
}

bool Tableau::operator==(Tableau& other) {
  if (num_qubits != other.num_qubits) {
    return false;
  }

  rref();
  other.rref();

  for (uint32_t i = 0; i < stabilizers.size(); i++) {
    if (stabilizers[i].get_r() != other.stabilizers[i].get_r()) {
      return false;
    }

    for (uint32_t j = 0; j < num_qubits; j++) {
      if (stabilizers[i].get_z(j) != other.stabilizers[i].get_z(j)) {
        return false;
      }

      if (stabilizers[i].get_x(j) != other.stabilizers[i].get_x(j)) {
        return false;
      }
    }
  }

  return true;
}

void Tableau::rref(const Qubits& sites) {
  uint32_t pivot_row = 0;
  uint32_t row = 0;

  for (uint32_t k = 0; k < 2*sites.size(); k++) {
    uint32_t c = sites[k % sites.size()];
    bool z = k < sites.size();
    bool found_pivot = false;
    for (uint32_t i = row; i < stabilizers.size(); i++) {
      if ((z && stabilizers[i].get_z(c)) || (!z && stabilizers[i].get_x(c))) {
        pivot_row = i;
        found_pivot = true;
        break;
      }
    }

    if (found_pivot) {
      std::swap(stabilizers[row], stabilizers[pivot_row]);
      std::swap(destabilizers[row], destabilizers[pivot_row]);

      for (uint32_t i = 0; i < stabilizers.size(); i++) {
        if (i == row) {
          continue;
        }

        if ((z && stabilizers[i].get_z(c)) || (!z && stabilizers[i].get_x(c))) {
          stabilizers[i] = stabilizers[i] * stabilizers[row];
          destabilizers[row] = destabilizers[row] * destabilizers[i];
        }
      }

      row += 1;
    } else {
      continue;
    }
  }
}

void Tableau::xrref(const Qubits& sites) {
  uint32_t pivot_row = 0;
  uint32_t row = 0;

  for (uint32_t k = 0; k < 2*sites.size(); k++) {
    uint32_t c = sites[k % sites.size()];
    bool z = k < sites.size();
    bool found_pivot = false;
    for (uint32_t i = row; i < stabilizers.size(); i++) {
      if (!z && stabilizers[i].get_x(c)) {
        pivot_row = i;
        found_pivot = true;
        break;
      }
    }

    if (found_pivot) {
      std::swap(stabilizers[row], stabilizers[pivot_row]);

      for (uint32_t i = 0; i < stabilizers.size(); i++) {
        if (i == row) {
          continue;
        }

        if (!z && stabilizers[i].get_x(c)) {
          stabilizers[i] = stabilizers[i] * stabilizers[row];
        }
      }

      row += 1;
    } else {
      continue;
    }
  }
}

uint32_t Tableau::xrank(const Qubits& sites) {
  xrref(sites);

  uint32_t r = 0;
  for (uint32_t i = 0; i < stabilizers.size(); i++) {
    for (uint32_t j = 0; j < sites.size(); j++) {
      if (stabilizers[i].get_x(sites[j])) {
        r++;
        break;
      }
    }
  }

  return r;
}

uint32_t Tableau::rank(const Qubits& sites) {
  rref(sites);

  uint32_t r = 0;
  for (uint32_t i = 0; i < stabilizers.size(); i++) {
    for (uint32_t j = 0; j < sites.size(); j++) {
      if (stabilizers[i].get_x(sites[j]) || stabilizers[i].get_z(sites[j])) {
        r++;
        break;
      }
    }
  }

  return r;
}

void Tableau::rref() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  rref(qubits);
}

void Tableau::xrref() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  xrref(qubits);
}

uint32_t Tableau::rank() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return rank(qubits);
}

uint32_t Tableau::xrank() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return xrank(qubits);
}

Tableau Tableau::partial_trace(const Qubits& qubits) {
  rref(qubits);

  Tableau tableau_new;
  tableau_new.num_qubits = num_qubits - qubits.size();

  Qubits qubits_complement = to_qubits(support_complement(qubits, num_qubits));

  for (const PauliString& stab : stabilizers) {
    bool is_id = true;
    for (size_t i = 0; i < qubits.size(); i++) {
      if (stab.get_x(qubits[i]) || stab.get_z(qubits[i])) {
        is_id = false;
        break;
      }
    }

    if (is_id) {
      tableau_new.stabilizers.push_back(stab.substring(qubits_complement, true));
    }
  }

  return tableau_new;
}

double Tableau::bitstring_amplitude(const BitString& bits) {
  if (bits.num_bits != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot evaluate a bitstring of {} bits on a Tableau of {} qubits.", bits.num_bits, num_qubits));
  }

  double p = 1/std::pow(2.0, xrank() + num_qubits - stabilizers.size());

  bool in_support = true;
  for (size_t r = 0; r < stabilizers.size(); r++) {
    const PauliString& row = stabilizers[r];

    // Need to check that every z-only stabilizer g acts on |z> as g|z> = |z>. 
    bool has_x = false;
    for (size_t i = 0; i < num_qubits; i++) {
      if (row.get_x(i)) {
        has_x = true;
        break;
      }
    }

    if (has_x) {
      continue;
    }

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

std::string Tableau::to_string(bool print_destabilizers) const {
  std::string s = "";
  if (print_destabilizers) {
    for (size_t i = 0; i < destabilizers.size(); i++) {
      s += (i == 0) ? "[" : " ";
      s += destabilizers[i].to_string();
      s += (i == destabilizers.size() - 1) ? "]" : "\n";
    }
    s += "\n";
  }

  for (size_t i = 0; i < stabilizers.size(); i++) {
    s += (i == 0) ? "[" : " ";
    s += stabilizers[i].to_string();
    s += (i == stabilizers.size() - 1) ? "]" : "\n";
  }

  return s;
}

std::string Tableau::to_string_ops(bool print_destabilizers) const {
  std::string s = "";
  if (print_destabilizers) {
    for (size_t i = 0; i < destabilizers.size(); i++) {
      s += (i == 0) ? "[" : " ";
      s += destabilizers[i].to_string_ops();
      s += (i == destabilizers.size() - 1) ? "]" : "\n";
    }
    s += "\n";
  }

  for (size_t i = 0; i < stabilizers.size(); i++) {
    s += (i == 0) ? "[" : " ";
    s += stabilizers[i].to_string_ops();
    s += (i == stabilizers.size() - 1) ? "]" : "\n";
  }
  return s;
}

void Tableau::h(uint32_t a) {
  validate_qubit(a);
  for (PauliString& p : stabilizers) {
    p.h(a);
  }

  for (PauliString& p : destabilizers) {
    p.h(a);
  }
}

void Tableau::s(uint32_t a) {
  validate_qubit(a);
  for (PauliString& p : stabilizers) {
    p.s(a);
  }

  for (PauliString& p : destabilizers) {
    p.s(a);
  }
}

void Tableau::sd(uint32_t a) {
  s(a);
  s(a);
  s(a);
}

void Tableau::x(uint32_t a) {
  validate_qubit(a);
  for (PauliString& p : stabilizers) {
    p.x(a);
  }

  for (PauliString& p : destabilizers) {
    p.x(a);
  }
}

void Tableau::y(uint32_t a) {
  validate_qubit(a);
  for (PauliString& p : stabilizers) {
    p.y(a);
  }

  for (PauliString& p : destabilizers) {
    p.y(a);
  }
}

void Tableau::z(uint32_t a) {
  validate_qubit(a);
  for (PauliString& p : stabilizers) {
    p.z(a);
  }

  for (PauliString& p : destabilizers) {
    p.z(a);
  }
}

void Tableau::cx(uint32_t a, uint32_t b) {
  validate_qubit(a);
  validate_qubit(b);
  for (PauliString& p : stabilizers) {
    p.cx(a, b);
  }

  for (PauliString& p : destabilizers) {
    p.cx(a, b);
  }
}

void Tableau::cz(uint32_t a, uint32_t b) {
  h(b);
  cx(a, b);
  h(b);
}

std::pair<bool, uint32_t> Tableau::mzr_deterministic(uint32_t a) const {
  for (uint32_t p = 0; p < stabilizers.size(); p++) {
    // Suitable p identified; outcome is random
    if (stabilizers[p].get_x(a)) { 
      return std::pair(false, p);
    }
  }

  // No p found; outcome is deterministic
  return std::pair(true, 0);
}

MeasurementData Tableau::mzr(uint32_t a, std::optional<bool> outcome) {
  validate_qubit(a);

  auto [deterministic, p] = mzr_deterministic(a);

  if (!deterministic) {
    bool b = outcome ? outcome.value() : randi() % 2;

    for (uint32_t i = 0; i < stabilizers.size(); i++) {
      if (i != p && stabilizers[i].get_x(a)) {
        stabilizers[i] = stabilizers[i] * stabilizers[p];
      }
    }

    for (uint32_t i = 0; i < destabilizers.size(); i++) {
      if (destabilizers[i].get_x(a)) {
        destabilizers[i] = destabilizers[i] * stabilizers[p];
      }
    }

    std::swap(stabilizers[p], destabilizers[p]);

    stabilizers[p] = PauliString(num_qubits);
    stabilizers[p].set_r(b ? 2 : 0);
    stabilizers[p].set_z(a, true);

    return {b, 0.5};
  } else { // deterministic
    PauliString scratch(num_qubits);
    for (uint32_t i = 0; i < stabilizers.size(); i++) {
      if (destabilizers[i].get_x(a)) {
        scratch = scratch * stabilizers[i];
      }
    }

    bool b = (scratch.get_r() == 2);

    if (outcome) {
      if (b != outcome.value()) {
        throw std::runtime_error("Invalid forced measurement of QuantumCHPState.");
      }
    }
    
    return {b, 1.0};
  }
}
