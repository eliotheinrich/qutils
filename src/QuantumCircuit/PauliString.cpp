#include "PauliString.hpp"
#include "Instructions.hpp"
#include "QuantumCircuit.h"

BitString::BitString(uint32_t num_bits) : num_bits(num_bits) {
  size_t width = num_bits / binary_word_size() + static_cast<bool>(num_bits % binary_word_size());
  bits = std::vector<binary_word>(width, 0);
}

binary_word BitString::to_integer() const {
  if (bits.size() > 1) {
    throw std::runtime_error(fmt::format("Cannot convert a bitstring containing more than {} bits.", binary_word_size()));
  }

  return bits[0];
}

BitString BitString::from_bits(size_t num_bits, binary_word bits) {
  if (num_bits >= binary_word_size()) {
    throw std::runtime_error(fmt::format("Cannot create a >{} BitString from a {}-bit integer.", binary_word_size(), sizeof(bits)));
  }

  BitString bit_string(num_bits);

  bit_string.bits = std::vector<binary_word>(1);
  bit_string[0] = bits;

  return bit_string;
}

BitString BitString::random(size_t num_bits, double p) {
  BitString bits(num_bits);

  for (size_t i = 0; i < num_bits; i++) {
    bool v = randf() < p;
    bits.set(i, v);
  }

  return bits;
}

uint32_t BitString::hamming_weight() const {
  uint32_t r = 0;
  for (size_t i = 0; i < num_bits; i++) {
    r += get(i);
  }
  return r;
}

QubitInterval BitString::support_range() const {
  uint32_t first = -1;
  uint32_t last = -1;

  for (size_t i = 0; i < num_bits; ++i) {
    if (get(i)) {
      if (first == -1) {
        first = i;
      }
      last = i;
    }
  }

  if (first == -1) {
    return std::nullopt;
  }

  return std::make_pair(first, last + 1);
}

uint32_t BitString::size() const {
  return bits.size();
}

const binary_word& BitString::operator[](uint32_t i) const {
  return bits[i];
}

binary_word& BitString::operator[](uint32_t i) {
  return bits[i];
}

BitString BitString::operator^(const BitString& other) const {
  if (size() != other.size()) {
    throw std::runtime_error(fmt::format("Tried to perform ^ on BitStrings of unequal length: {} and {}", size(), other.size()));
  }

  BitString new_bits(num_bits);

  for (size_t i = 0; i < size(); i++) {
    new_bits[i] = bits[i] ^ other.bits[i];
  }

  return new_bits;
}

BitString& BitString::operator^=(const BitString& other) {
  if (size() != other.size()) {
    throw std::runtime_error(fmt::format("Tried to perform ^ on BitStrings of unequal length: {} and {}", size(), other.size()));
  }

  for (size_t i = 0; i < bits.size(); ++i) {
    bits[i] ^= other.bits[i];
  }

  return *this;
}

BitString BitString::substring(const std::vector<uint32_t>& kept_bits, bool remove_bits) const {
  size_t n = remove_bits ? kept_bits.size() : num_bits;
  BitString b(n);

  if (remove_bits) {
    for (size_t i = 0; i < kept_bits.size(); i++) {
      b.set(i, get(kept_bits[i]));
    }
  } else {
    for (const auto i : kept_bits) {
      b.set(i, get(i));
    }
  }

  return b;
}

BitString BitString::superstring(const std::vector<uint32_t>& sites, size_t new_num_bits) const {
  if (sites.size() != num_bits) {
    throw std::runtime_error(fmt::format("When constructing a superstring bitstring, provided sites must have same size as num_qubits."));
  }
  BitString b(new_num_bits);

  for (size_t i = 0; i < sites.size(); i++) {
    b.set(sites[i], get(i));
  }

  return b;
}

PauliString::PauliString(uint32_t num_qubits) : num_qubits(num_qubits), phase(0) {
  if (num_qubits == 0) {
    throw std::runtime_error("Cannot create a 0-qubit PauliString.");
  }
  bit_string = BitString(2u * num_qubits);
}

PauliString::PauliString(const PauliString& other) {
  num_qubits = other.num_qubits;
  bit_string = other.bit_string;
  phase = other.phase;
}

uint32_t PauliString::process_pauli_string(const std::string& paulis) {
  uint32_t num_qubits = paulis.size();
  if (paulis[0] == '+' || paulis[0] == '-') {
    num_qubits--;

    if (paulis[1] == 'i') {
      num_qubits--;
    }
  }
  return num_qubits;
}

PauliString::PauliString(const std::string& paulis) : PauliString(process_pauli_string(paulis)) {
  std::string s = paulis;
  phase = parse_phase(s);

  for (size_t i = 0; i < num_qubits; i++) {
    if (s[i] == 'I') {
      set_x(i, false);
      set_z(i, false);
    } else if (s[i] == 'X') {
      set_x(i, true);
      set_z(i, false);
    } else if (s[i] == 'Y') {
      set_x(i, true);
      set_z(i, true);
    } else if (s[i] == 'Z') {
      set_x(i, false);
      set_z(i, true);
    } else {
      std::cout << fmt::format("character {} not recognized\n", s[i]);
      throw std::runtime_error(fmt::format("Invalid string {} used to create PauliString.", paulis));
    }
  }
}

PauliString::PauliString(const std::vector<Pauli>& paulis, uint8_t phase) : PauliString(paulis.size()) { 
  for (size_t i = 0; i < paulis.size(); i++) {
    set_op(i, paulis[i]);
  }

  set_r(phase);
}

PauliString PauliString::rand(uint32_t num_qubits) {
  PauliString p(num_qubits);

  for (size_t i = 0; i < p.bit_string.size(); i++) {
    p.bit_string[i] = randi();
  }

  p.set_r(randi() % 4);

  if (num_qubits == 0) {
    return p;
  }

  // Need to check that at least one bit is nonzero so that p is not the identity
  for (uint32_t j = 0; j < num_qubits; j++) {
    if (p.get_xz(j)) {
      return p;
    }
  }

  return PauliString::rand(num_qubits);
}

PauliString PauliString::randh(uint32_t num_qubits) {
  PauliString p = PauliString::rand(num_qubits);
  p.set_r(randi(0, 2) * 2);

  return p;
}

PauliString PauliString::basis(uint32_t num_qubits, const std::string& P, uint32_t q, uint8_t r) {
  PauliString p(num_qubits);
  if (P == "X") {
    p.set_x(q, true);
  } else if (P == "Y") {
    p.set_x(q, true);
    p.set_z(q, true);
  } else if (P == "Z") {
    p.set_z(q, true);
  } else {
    std::string error_message = P + " is not a valid basis. Must provide one of X,Y,Z.\n";
    throw std::invalid_argument(error_message);
  }

  p.set_r(r);

  return p;
}

PauliString PauliString::from_bitstring(uint32_t num_qubits, uint32_t bits) {
  PauliString p = PauliString(num_qubits);
  p.bit_string = BitString::from_bits(2u * num_qubits, bits);
  return p;
}

PauliString PauliString::basis(uint32_t num_qubits, const std::string& P, uint32_t q) {
  return PauliString::basis(num_qubits, P, q, false);
}

PauliString PauliString::substring(const Qubits& qubits, bool remove_qubits) const {
  size_t n = remove_qubits ? qubits.size() : num_qubits;
  PauliString p(n);

  if (remove_qubits) {
    for (size_t i = 0; i < qubits.size(); i++) {
      p.set_x(i, get_x(qubits[i]));
      p.set_z(i, get_z(qubits[i]));
    }
  } else {
    for (const auto q : qubits) {
      p.set_x(q, get_x(q));
      p.set_z(q, get_z(q));
    }
  }

  p.phase = phase;
  return p;
}

PauliString PauliString::substring(const QubitSupport& support, bool remove_qubits) const {
  return substring(to_qubits(support), remove_qubits);
}

PauliString PauliString::superstring(const Qubits& qubits, size_t new_num_qubits) const {
  if (qubits.size() != num_qubits) {
    throw std::runtime_error(fmt::format("When constructing a superstring Pauli, provided sites must have same size as num_qubits. P = {}, qubits = {}.", to_string_ops(), qubits));
  }
  std::vector<Pauli> paulis(new_num_qubits, Pauli::I);
  for (size_t i = 0; i < num_qubits; i++) {
    uint32_t q = qubits[i];

    paulis[q] = to_pauli(i);
  }

  return PauliString(paulis, get_r());
}

uint8_t PauliString::get_multiplication_phase(const PauliString& p1, const PauliString& p2) {
  uint8_t s = p1.get_r() + p2.get_r();

  for (uint32_t j = 0; j < p1.num_qubits; j++) {
    s += multiplication_phase(p1.get_xz(j), p2.get_xz(j));
  }

  return s;
}

bool PauliString::hermitian() const {
  return !(phase & 0b1); // mod 2
}

bool PauliString::is_basis() const {
  Qubits support = get_support();
  if (support.size() != 1) {
    return false;
  } 

  uint32_t q = support[0];
  return get_z(q) && !get_x(q) && hermitian();
}

PauliString PauliString::operator*(const PauliString& other) const {
  if (num_qubits != other.num_qubits) {
    throw std::runtime_error(fmt::format("Multiplying PauliStrings with {} qubits and {} qubits do not match.", num_qubits, other.num_qubits));
  }

  PauliString p(num_qubits);

  p.set_r(PauliString::get_multiplication_phase(*this, other));

  p.bit_string = bit_string ^ other.bit_string;

  return p;
}

PauliString PauliString::operator-() { 
  set_r(get_r() + 2);
  return *this;
}

bool PauliString::operator==(const PauliString &rhs) const {
  if (num_qubits != rhs.num_qubits) {
    return false;
  }

  if (get_r() != rhs.get_r()) {
    return false;
  }

  for (uint32_t i = 0; i < num_qubits; i++) {
    if (get_x(i) != rhs.get_x(i)) {
      return false;
    }

    if (get_z(i) != rhs.get_z(i)) {
      return false;
    }
  }

  return true;
}

bool PauliString::operator!=(const PauliString &rhs) const { 
  return !(this->operator==(rhs)); 
}

Eigen::Matrix2cd PauliString::to_matrix(uint32_t i) const {
  std::string s = to_op(i);

  Eigen::Matrix2cd g;
  if (s == "I") {
    g = gates::I::value.asDiagonal();
  } else if (s == "X") {
    g << gates::X::value;
  } else if (s == "Y") {
    g = gates::Y::value;
  } else {
    g = gates::Z::value.asDiagonal();
  }

  return g;
}

Eigen::MatrixXcd PauliString::to_matrix() const {
  Eigen::MatrixXcd g = to_matrix(0);

  for (uint32_t i = 1; i < num_qubits; i++) {
    Eigen::MatrixXcd gi = to_matrix(i);
    Eigen::MatrixXcd g0 = g;
    g = Eigen::kroneckerProduct(gi, g0);
  }

  constexpr std::complex<double> i(0.0, 1.0);

  if (phase == 1) {
    g = i*g;
  } else if (phase == 2) {
    g = -g;
  } else if (phase == 3) {
    g = -i*g;
  }

  return g;
}

Pauli PauliString::to_pauli(uint32_t i) const {
  bool xi = get_x(i);
  bool zi = get_z(i);

  if (xi && zi) {
    return Pauli::Y;
  } else if (!xi && zi) {
    return Pauli::Z;
  } else if (xi && !zi) {
    return Pauli::X;
  } else {
    return Pauli::I;
  }
}

std::vector<Pauli> PauliString::to_pauli() const {
  std::vector<Pauli> paulis(num_qubits);
  std::generate(paulis.begin(), paulis.end(), [n = 0, this]() mutable { return to_pauli(n++); });
  return paulis;
}

QubitInterval PauliString::support_range() const {
  std::vector<Pauli> paulis = to_pauli();
  auto first = std::ranges::find_if(paulis, [&](Pauli pi) { return pi != Pauli::I; });
  auto last = std::ranges::find_if(paulis | std::views::reverse, [&](Pauli pi) { return pi != Pauli::I; });

  if (first == paulis.end() && last == paulis.rend()) {
    return std::nullopt;
  } else {
    uint32_t q1 = std::distance(paulis.begin(), first);
    uint32_t q2 = num_qubits - std::distance(paulis.rbegin(), last);
    return std::make_pair(q1, q2);
  }
}

Qubits PauliString::get_support() const {
  Qubits support;
  for (size_t i = 0; i < num_qubits; i++) {
    if (to_pauli(i) != Pauli::I) {
      support.push_back(i);
    }
  }

  return support;
}

std::string PauliString::to_op(uint32_t i) const {
  bool xi = get_x(i); 
  bool zi = get_z(i);

  if (xi && zi) {
    return "Y";
  } else if (!xi && zi) {
    return "Z";
  } else if (xi && !zi) {
    return "X";
  } else {
    return "I";
  }
}

std::string PauliString::to_string() const {
  std::string s = "[ ";
  for (uint32_t i = 0; i < num_qubits; i++) {
    s += get_x(i) ? "1" : "0";
  }

  for (uint32_t i = 0; i < num_qubits; i++) {
    s += get_z(i) ? "1" : "0";
  }

  s += " | ";
  s += phase_to_string(phase);
  s += " ]";

  return s;
}

std::string PauliString::to_string_ops() const {
  std::string s = phase_to_string(phase);

  for (uint32_t i = 0; i < num_qubits; i++) {
    s += to_op(i);
  }

  return s;
}

void PauliString::evolve(const QuantumCircuit& qc) {
  if (!qc.is_clifford()) {
    throw std::runtime_error("Provided circuit is not Clifford.");
  }

  if (qc.get_num_qubits() != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot evolve a Paulistring with {} qubits with a QuantumCircuit with {} qubits.", num_qubits, qc.get_num_qubits()));
  }

  auto apply_qinst = [this](const QuantumInstruction& qinst) {
    std::visit(quantumcircuit_utils::overloaded{
      [this](std::shared_ptr<Gate> gate) { 
        std::string name = gate->label();

        if (name == "H") {
          h(gate->qubits[0]);
        } else if (name == "S") {
          s(gate->qubits[0]);
        } else if (name == "Sd") {
          sd(gate->qubits[0]);
        } else if (name == "X") {
          x(gate->qubits[0]);
        } else if (name == "Y") {
          y(gate->qubits[0]);
        } else if (name == "Z") {
          z(gate->qubits[0]);
        } else if (name == "sqrtX") {
          sqrtX(gate->qubits[0]);
        } else if (name == "sqrtY") {
          sqrtY(gate->qubits[0]);
        } else if (name == "sqrtZ") {
          sqrtZ(gate->qubits[0]);
        } else if (name == "sqrtXd") {
          sqrtXd(gate->qubits[0]);
        } else if (name == "sqrtYd") {
          sqrtYd(gate->qubits[0]);
        } else if (name == "sqrtZd") {
          sqrtZd(gate->qubits[0]);
        } else if (name == "CX") {
          cx(gate->qubits[0], gate->qubits[1]);
        } else if (name == "CY") {
          cy(gate->qubits[0], gate->qubits[1]);
        } else if (name == "CZ") {
          cz(gate->qubits[0], gate->qubits[1]);
        } else if (name == "SWAP") {
          swap(gate->qubits[0], gate->qubits[1]);
        } else {
          throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to PauliString.evolve.", name));
        }
      },
      [](const FreeFermionGate& gate) {
        throw std::runtime_error("Cannot evolve arbitrary fermionic gate on PauliString.");
      },
      [](const Measurement& m) { 
        throw std::runtime_error("Cannot do measure on a single PauliString.");
      },
      [](const WeakMeasurement& m) { 
        throw std::runtime_error("Cannot do weak measurement on a single PauliString.");
      },
    }, qinst);
  };

  for (auto const &inst : qc.instructions) {
    std::visit(quantumcircuit_utils::overloaded{
      [&apply_qinst](const QuantumInstruction& qinst) {
        apply_qinst(qinst);
      },
      [](const ClassicalInstruction& clinst) {
        throw std::runtime_error("Cannot do classical operation on PauliString.");
      },
      [&apply_qinst](const ConditionedInstruction& cinst) {
        if (cinst.control || cinst.target) {
          throw std::runtime_error("Cannot do classically-conditioned operation on PauliString.");
        }
        apply_qinst(cinst.inst);
      }
    }, inst);
  }
}

QuantumCircuit PauliString::transform(PauliString const &p) const {
  Qubits qubits(p.num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);

  QuantumCircuit qc1(p.num_qubits);
  reduce(true, std::make_pair(&qc1, qubits));

  QuantumCircuit qc2(p.num_qubits);
  p.reduce(true, std::make_pair(&qc2, qubits));

  qc1.append(qc2.adjoint());

  return qc1;
}


void PauliString::s(uint32_t a) {
  uint8_t xza = get_xz(a);
  bool xa = (xza >> 0u) & 1u;
  bool za = (xza >> 1u) & 1u;

  uint8_t r = phase;

  constexpr uint8_t s_phase_lookup[] = {0, 0, 0, 2};
  set_r(r + s_phase_lookup[xza]);
  set_z(a, xa != za);
}

void PauliString::sd(uint32_t a) {
  s(a);
  s(a);
  s(a);
}

void PauliString::h(uint32_t a) {
  uint8_t xza = get_xz(a);
  bool xa = (xza >> 0u) & 1u;
  bool za = (xza >> 1u) & 1u;

  uint8_t r = phase;

  constexpr uint8_t h_phase_lookup[] = {0, 0, 0, 2};
  set_r(r + h_phase_lookup[xza]);
  set_x(a, za);
  set_z(a, xa);
}

void PauliString::x(uint32_t a) {
  h(a);
  s(a);
  s(a);
  h(a);
}

void PauliString::y(uint32_t a) {
  h(a);
  s(a);
  s(a);
  h(a);
  s(a);
  s(a);
}

void PauliString::z(uint32_t a) {
  s(a);
  s(a);
}

void PauliString::sqrtX(uint32_t a) {
  sd(a);
  h(a);
  sd(a);
}

void PauliString::sqrtXd(uint32_t a) {
  s(a);
  h(a);
  s(a);
}

void PauliString::sqrtY(uint32_t a) {
  z(a);
  h(a);
}

void PauliString::sqrtYd(uint32_t a) {
  h(a);
  z(a);
}

void PauliString::sqrtZ(uint32_t a) {
  s(a);
}

void PauliString::sqrtZd(uint32_t a) {
  sd(a);
}

void PauliString::cx(uint32_t a, uint32_t b) {
  uint8_t xza = get_xz(a);
  bool xa = (xza >> 0u) & 1u;
  bool za = (xza >> 1u) & 1u;

  uint8_t xzb = get_xz(b);
  bool xb = (xzb >> 0u) & 1u;
  bool zb = (xzb >> 1u) & 1u;

  uint8_t bitcode = xzb + (xza << 2);

  uint8_t r = phase;

  constexpr uint8_t cx_phase_lookup[] = {0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2};
  set_r(r + cx_phase_lookup[bitcode]);
  set_x(b, xa != xb);
  set_z(a, za != zb);
}

void PauliString::cy(uint32_t a, uint32_t b) {
  s(a);
  h(a);
  cz(b, a);
  h(a);
  s(a);
  s(a);
  s(a);
}

void PauliString::cz(uint32_t a, uint32_t b) {
  h(b);
  cx(a, b);
  h(b);
}

void PauliString::swap(uint32_t a, uint32_t b) {
  cx(a, b);
  cx(b, a);
  cx(a, b);
}

bool PauliString::commutes_at(const PauliString& p, uint32_t i) const {
  if ((get_x(i) == p.get_x(i)) && (get_z(i) == p.get_z(i))) { // operators are identical
    return true;
  } else if (!get_x(i) && !get_z(i)) { // this is identity
    return true;
  } else if (!p.get_x(i) && !p.get_z(i)) { // other is identity
    return true;
  } else {
    return false; 
  }
}

bool PauliString::commutes(const PauliString& p) const {
  if (num_qubits != p.num_qubits) {
    throw std::invalid_argument(fmt::format("p = {} has {} qubits and q = {} has {} qubits; cannot check commutation.", p.to_string_ops(), p.num_qubits, to_string_ops(), num_qubits));
  }

  uint32_t anticommuting_indices = 0u;
  for (uint32_t i = 0; i < num_qubits; i++) {
    if (!commutes_at(p, i)) {
      anticommuting_indices++;
    }
  }

  return anticommuting_indices % 2 == 0;
}
