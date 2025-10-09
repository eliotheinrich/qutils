#include "Instructions.hpp"
#include "CircuitUtils.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <set>

#include <fmt/format.h>
#include <fmt/ranges.h>

const std::unordered_map<SymbolicGate::GateLabel, Eigen::MatrixXcd> SymbolicGate::gate_map = {
  { SymbolicGate::GateLabel::H,      process_gate_data(gates::H::value)},
  { SymbolicGate::GateLabel::X,      process_gate_data(gates::X::value)},
  { SymbolicGate::GateLabel::Y,      process_gate_data(gates::Y::value)},
  { SymbolicGate::GateLabel::Z,      process_gate_data(gates::Z::value)},
  { SymbolicGate::GateLabel::sqrtX,  process_gate_data(gates::sqrtX::value)},
  { SymbolicGate::GateLabel::sqrtY,  process_gate_data(gates::sqrtY::value)},
  { SymbolicGate::GateLabel::S,      process_gate_data(gates::sqrtZ::value)},
  { SymbolicGate::GateLabel::sqrtXd, process_gate_data(gates::sqrtXd::value)},
  { SymbolicGate::GateLabel::sqrtYd, process_gate_data(gates::sqrtYd::value)},
  { SymbolicGate::GateLabel::Sd,     process_gate_data(gates::sqrtZd::value)},
  { SymbolicGate::GateLabel::T,      process_gate_data(gates::T::value)},
  { SymbolicGate::GateLabel::Td,     process_gate_data(gates::Td::value)},
  { SymbolicGate::GateLabel::CX,     process_gate_data(gates::CX::value)},
  { SymbolicGate::GateLabel::CY,     process_gate_data(gates::CY::value)},
  { SymbolicGate::GateLabel::CZ,     process_gate_data(gates::CZ::value)},
  { SymbolicGate::GateLabel::SWAP,   process_gate_data(gates::SWAP::value)},
};

bool SymbolicGate::str_equal_ci(const char* a, const char* b) {
  while (*a && *b) {
    if (std::tolower(*a) != std::tolower(*b)) {
      return false;
    }
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

SymbolicGate::GateLabel SymbolicGate::parse_gate(const char* name) {
  if (str_equal_ci(name, "h")) {
    return SymbolicGate::GateLabel::H;
  } else if (str_equal_ci(name, "x")) {
    return SymbolicGate::GateLabel::X;
  } else if (str_equal_ci(name, "y")) {
    return SymbolicGate::GateLabel::Y;
  } else if (str_equal_ci(name, "z")) {
    return SymbolicGate::GateLabel::Z;
  } else if (str_equal_ci(name, "sqrtx")) {
    return SymbolicGate::GateLabel::sqrtX;
  } else if (str_equal_ci(name, "sqrty")) {
    return SymbolicGate::GateLabel::sqrtY;
  } else if (str_equal_ci(name, "sqrtz") || str_equal_ci(name, "s")) {
    return SymbolicGate::GateLabel::S;
  } else if (str_equal_ci(name, "sqrtxd")) {
    return SymbolicGate::GateLabel::sqrtXd;
  } else if (str_equal_ci(name, "sqrtyd")) {
    return SymbolicGate::GateLabel::sqrtYd;
  } else if (str_equal_ci(name, "sqrtzd") || str_equal_ci(name, "sd")) {
    return SymbolicGate::GateLabel::Sd;
  } else if (str_equal_ci(name, "t")) {
    return SymbolicGate::GateLabel::T;
  } else if (str_equal_ci(name, "td")) {
    return SymbolicGate::GateLabel::Td;
  } else if (str_equal_ci(name, "cx")) {
    return SymbolicGate::GateLabel::CX;
  } else if (str_equal_ci(name, "cy")) {
    return SymbolicGate::GateLabel::CY;
  } else if (str_equal_ci(name, "cz")) {
    return SymbolicGate::GateLabel::CZ;
  } else if (str_equal_ci(name, "swap")) {
    return SymbolicGate::GateLabel::SWAP;
  } else {
    throw std::runtime_error(fmt::format("Error: unknown gate {}.", name));
  }
}

const char* SymbolicGate::type_to_string(SymbolicGate::GateLabel g) {
  switch (g) {
    case SymbolicGate::GateLabel::H:
      return "H";
    case SymbolicGate::GateLabel::X:
      return "X";
    case SymbolicGate::GateLabel::Y:
      return "Y";
    case SymbolicGate::GateLabel::Z:
      return "Z";
    case SymbolicGate::GateLabel::sqrtX:
      return "sqrtX";
    case SymbolicGate::GateLabel::sqrtY:
      return "sqrtY";
    case SymbolicGate::GateLabel::S:
      return "S";
    case SymbolicGate::GateLabel::sqrtXd:
      return "sqrtXd";
    case SymbolicGate::GateLabel::sqrtYd:
      return "sqrtYd";
    case SymbolicGate::GateLabel::Sd:
      return "Sd";
    case SymbolicGate::GateLabel::T:
      return "T";
    case SymbolicGate::GateLabel::Td:
      return "Td";
    case SymbolicGate::GateLabel::CX:
      return "CX";
    case SymbolicGate::GateLabel::CY:
      return "CY";
    case SymbolicGate::GateLabel::CZ:
      return "CZ";
    case SymbolicGate::GateLabel::SWAP:
      return "SWAP";
    default:
      throw std::runtime_error("Invalid gate type.");
  }
}

size_t SymbolicGate::num_qubits_for_gate(SymbolicGate::GateLabel g) {
  switch (g) {
    case SymbolicGate::GateLabel::H:
      return 1;
    case SymbolicGate::GateLabel::X:
      return 1;
    case SymbolicGate::GateLabel::Y:
      return 1;
    case SymbolicGate::GateLabel::Z:
      return 1;
    case SymbolicGate::GateLabel::sqrtX:
      return 1;
    case SymbolicGate::GateLabel::sqrtY:
      return 1;
    case SymbolicGate::GateLabel::S:
      return 1;
    case SymbolicGate::GateLabel::sqrtXd:
      return 1;
    case SymbolicGate::GateLabel::sqrtYd:
      return 1;
    case SymbolicGate::GateLabel::Sd:
      return 1;
    case SymbolicGate::GateLabel::T:
      return 1;
    case SymbolicGate::GateLabel::Td:
      return 1;
    case SymbolicGate::GateLabel::CX:
      return 2;
    case SymbolicGate::GateLabel::CY:
      return 2;
    case SymbolicGate::GateLabel::CZ:
      return 2;
    case SymbolicGate::GateLabel::SWAP:
      return 2;
    default:
      throw std::runtime_error("Invalid gate type.");
  }
}

Eigen::MatrixXcd SymbolicGate::process_gate_data(const Eigen::MatrixXcd& data) {
  if (data.rows() == data.cols()) {
    return data;
  } else {
    return data.asDiagonal();
  }
}

bool SymbolicGate::is_clifford() const {
  return SymbolicGate::clifford_gates.contains(type);
}

uint32_t SymbolicGate::num_params() const {
  return 0;
}

std::string SymbolicGate::label() const {
  return type_to_string(type);
}

Eigen::MatrixXcd SymbolicGate::define(const std::vector<double>& params) const {
  if (params.size() != 0) {
    throw std::invalid_argument("Cannot pass parameters to SymbolicGate.");
  }

  return gate_map.at(type);
}

std::shared_ptr<Gate> SymbolicGate::adjoint() const {
  SymbolicGate::GateLabel new_type;
  if (SymbolicGate::adjoint_map.contains(type)) {
    new_type = SymbolicGate::adjoint_map[type];
  } else {
    new_type = type;
  }

  return std::shared_ptr<Gate>(new SymbolicGate(new_type, qubits));
}

std::shared_ptr<Gate> SymbolicGate::clone() {
  return std::shared_ptr<Gate>(new SymbolicGate(type, qubits)); 
}

uint32_t MatrixGate::num_params() const {
  return 0;
}

std::string MatrixGate::label() const {
  return label_str;
}

Eigen::MatrixXcd MatrixGate::define(const std::vector<double>& params) const {
  if (params.size() != 0) {
    throw std::invalid_argument("Cannot pass parameters to MatrixGate.");
  }

  return data;
}

std::shared_ptr<Gate> MatrixGate::adjoint() const {
  return std::shared_ptr<Gate>(new MatrixGate(data.adjoint(), qubits));
}

bool MatrixGate::is_clifford() const {
  // No way to easily check if arbitrary data is Clifford at the moment
  return false;
}

std::shared_ptr<Gate> MatrixGate::clone() { 
  return std::shared_ptr<Gate>(new MatrixGate(data, qubits)); 
}

uint32_t PauliRotationGate::num_params() const {
  return 1;
}

std::string PauliRotationGate::label() const {
  return fmt::format("R({}){}", pauli, adj ? "d" : "");
}

Eigen::MatrixXcd PauliRotationGate::define(const std::vector<double>& params) const {
  if (params.size() != num_params()) {
    std::string error_message = "Invalid number of params passed to define(). Expected " 
      + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
    throw std::invalid_argument(error_message);
  }

  double t = params[0]/2;
  Eigen::MatrixXcd gate = (std::complex<double>(0.0, -t) * pauli.to_matrix()).exp();

  if (adj) {
    gate = gate.adjoint();
  }

  return gate;
}

bool PauliRotationGate::is_clifford() const {
  return false; 
}

std::shared_ptr<Gate> PauliRotationGate::adjoint() const {
  return std::make_shared<PauliRotationGate>(qubits, pauli, !adj);
}

std::shared_ptr<Gate> PauliRotationGate::clone() {
  return std::make_shared<PauliRotationGate>(qubits, pauli, adj);
}

uint32_t RxRotationGate::num_params() const {
  return 1;
}

std::string RxRotationGate::label() const { 
  return adj ? "Rxd" : "Rx";
}

Eigen::MatrixXcd RxRotationGate::define(const std::vector<double>& params) const {
  if (params.size() != num_params()) {
    std::string error_message = "Invalid number of params passed to define(). Expected " 
      + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
    throw std::invalid_argument(error_message);
  }

  Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(2, 2);

  double t = params[0]/2;
  gate << std::complex<double>(std::cos(t), 0),  std::complex<double>(0, -std::sin(t)), 
          std::complex<double>(0, -std::sin(t)), std::complex<double>(std::cos(t), 0);

  if (adj) {
    gate = gate.adjoint();
  }

  return gate;
}

bool RxRotationGate::is_clifford() const {
  return false;
}

std::shared_ptr<Gate> RxRotationGate::adjoint() const {
  return std::shared_ptr<Gate>(new RxRotationGate(qubits, !adj));
}

std::shared_ptr<Gate> RxRotationGate::clone() {
  return std::shared_ptr<Gate>(new RxRotationGate(qubits, adj));
}

uint32_t RyRotationGate::num_params() const { 
  return 1; 
}
    
std::string RyRotationGate::label() const { 
  return adj ? "Ryd" : "Ry";
}

Eigen::MatrixXcd RyRotationGate::define(const std::vector<double>& params) const {
  if (params.size() != num_params()) {
    std::string error_message = "Invalid number of params passed to define(). Expected " 
      + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
    throw std::invalid_argument(error_message);
  }

  Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(2, 2);

  double t = params[0]/2;
  gate << std::complex<double>(std::cos(t), 0), std::complex<double>(-std::sin(t), 0), 
          std::complex<double>(std::sin(t), 0), std::complex<double>( std::cos(t), 0);

  if (adj) {
    gate = gate.adjoint();
  }

  return gate;
}

bool RyRotationGate::is_clifford() const {
  return false;
}

std::shared_ptr<Gate> RyRotationGate::adjoint() const {
  return std::shared_ptr<Gate>(new RyRotationGate(qubits, !adj));
}

std::shared_ptr<Gate> RyRotationGate::clone() {
  return std::shared_ptr<Gate>(new RyRotationGate(qubits, adj));
}

uint32_t RzRotationGate::num_params() const {
  return 1; 
}

std::string RzRotationGate::label() const {
  return adj ? "Rzd" : "Rz"; 
}

Eigen::MatrixXcd RzRotationGate::define(const std::vector<double>& params) const {
  if (params.size() != num_params()) {
    std::string error_message = "Invalid number of params passed to define(). Expected " 
      + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
    throw std::invalid_argument(error_message);
  }

  Eigen::MatrixXcd gate = Eigen::MatrixXcd::Zero(2, 2);

  double t = params[0]/2;
  gate << std::complex<double>(std::cos(t), -std::sin(t)), std::complex<double>(0.0, 0.0), 
          std::complex<double>(0.0, 0.0), std::complex<double>(std::cos(t), std::sin(t));

  if (adj) {
    gate = gate.adjoint();
  }

  return gate;
}

bool RzRotationGate::is_clifford() const {
  return false;
}

std::shared_ptr<Gate> RzRotationGate::adjoint() const {
  return std::shared_ptr<Gate>(new RzRotationGate(qubits, !adj));
}

std::shared_ptr<Gate> RzRotationGate::clone() {
  return std::shared_ptr<Gate>(new RzRotationGate(qubits, adj));
}

std::shared_ptr<Gate> parse_gate(const std::string& s, const Qubits& qubits) {
  if (s == "H" || s == "h") {
    return std::make_shared<MatrixGate>(gates::H::value, qubits, "h");
  } else if (s == "X" || s == "x") {
    return std::make_shared<MatrixGate>(gates::X::value, qubits, "x");
  } else if (s == "Y" || s == "y") {
    return std::make_shared<MatrixGate>(gates::Y::value, qubits, "y");
  } else if (s == "Z" || s == "z") {
    return std::make_shared<MatrixGate>(gates::Z::value, qubits, "z");
  } else if (s == "RX" || s == "Rx" || s == "rx") {
    return std::make_shared<RxRotationGate>(qubits);
  } else if (s == "RXM" || s == "Rxm" || s == "rxm") {
    return std::make_shared<MemoizedGate<RxRotationGate>>(qubits);
  } else if (s == "RY" || s == "Ry" || s == "ry") {
    return std::make_shared<RyRotationGate>(qubits);
  } else if (s == "RYM" || s == "Rym" || s == "rym") {
    return std::make_shared<MemoizedGate<RyRotationGate>>(qubits);
  } else if (s == "RZ" || s == "Rz" || s == "rz") {
    return std::make_shared<RzRotationGate>(qubits);
  } else if (s == "RZM" || s == "Rzm" || s == "rzm") {
    return std::make_shared<MemoizedGate<RzRotationGate>>(qubits);
  } else if (s == "CX" || s == "cx") {
    return std::make_shared<MatrixGate>(gates::CX::value, qubits, "cx");
  } else if (s == "CY" || s == "cy") {
    return std::make_shared<MatrixGate>(gates::CY::value, qubits, "cy");
  } else if (s == "CZ" || s == "cz") {
    return std::make_shared<MatrixGate>(gates::CZ::value, qubits, "cz");
  } else if (s == "swap" || s == "SWAP") {
    return std::make_shared<MatrixGate>(gates::SWAP::value, qubits, "swap");
  } else {
    throw std::invalid_argument(fmt::format("Invalid gate type: {}", s));
  }
}

Eigen::MatrixXcd fermion_operator(size_t k, size_t num_qubits) {
  size_t basis = 1u << num_qubits;
  Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(basis, basis);

  for (size_t z = 0; z < basis; z++) {
    if (((z >> k) & 1) == 0) {
      size_t target = z | (1u << k);

      size_t parity = 0;
      for (size_t j = 0; j < k; ++j) {
        if ((z >> j) & 1) {
          parity++;
        }
      }
      std::complex<double> phase = (parity % 2 == 0) ? 1.0 : -1.0;

      M(target, z) = phase;
    }
  }

  return M;
}

PauliString majorana_operator(size_t k, size_t num_qubits) {
  PauliString P(num_qubits);
  size_t j = k/2;
  for (size_t i = 0; i < j; i++) {
    P.set_z(i, 1);
  }

  P.set_x(j, 1);
  if (k % 2) {
    P.set_z(j, 1);
  }

  return P;
}

void MajoranaGate::add_term(uint32_t i, uint32_t j, double a) {
  uint32_t i1 = std::min(i, j);
  uint32_t i2 = std::max(i, j);
  double sign = (i1 != i && !adj) ? -1.0 : 1.0;
  double amplitude = sign * a;

  auto idx = std::make_tuple(i1, i2);
  if (!term_map.contains(idx)) {
    term_map[idx] = terms.size();
    terms.push_back({i1, i2, 0.0});
  }

  int k = term_map[idx];
  terms[k].a += amplitude;
}

MajoranaGate MajoranaGate::combine(const MajoranaGate& other) const {
  if (!t || !other.t) {
    throw std::runtime_error("Cannot combine MajoranaGates with unbound t.");
  }

  MajoranaGate gate(*this);
  for (auto& term : gate.terms) {
    term.a *= gate.t.value();
  }

  for (const auto& term : other.terms) {
    gate.add_term(term.i, term.j, other.t.value()*term.a);
  }

  gate.set_t(1.0);
  return gate;
}

void MajoranaGate::apply_qubit_map(const Qubits& qubits) {
  for (auto& term : terms) {
    term.i = 2*qubits[term.i / 2] + term.i % 2;
    term.j = 2*qubits[term.j / 2] + term.j % 2;
  }
}

Eigen::MatrixXcd MajoranaGate::to_matrix() const {
  auto gate = to_gate();
  auto m = gate->define();
  return embed_unitary(gate->define(), gate->qubits, num_qubits);
}

Eigen::MatrixXcd term_to_matrix(const QuadraticMajoranaTerm& term) {
  int n = std::max(term.i, term.j)/2 + 1;
  PauliString p1 = majorana_operator(term.i, n);
  PauliString p2 = majorana_operator(term.j, n);

  return term.a * (p1 * p2).to_matrix();
}

QubitInterval get_term_support(const QuadraticMajoranaTerm& term) {
  uint32_t i = std::min(term.i, term.j)/2;
  uint32_t j = std::max(term.i, term.j)/2;

  return std::make_pair(i, j+1);
}

Qubits MajoranaGate::get_support() const {
  std::set<uint32_t> support;

  // Get total support
  for (const auto& term : terms) {
    Qubits term_support = to_qubits(get_term_support(term));
    for (auto q : term_support) {
      support.insert(q);
    }
  }

  return Qubits(support.begin(), support.end());
}

std::shared_ptr<Gate> MajoranaGate::to_gate() const {
  if (!t) {
    throw std::runtime_error("Cannot convert a FreeFermionGate with unbound parameter to a matrix. Call bind_parameters([t]) first.");
  }

  Qubits support = get_support();
  MajoranaGate gate(*this);
  Qubits map = reduced_support(support, num_qubits);
  gate.apply_qubit_map(map);

  size_t N = support.size();
  Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(1u << N, 1u << N);
  for (const auto& term : gate.terms) {
    H += embed_unitary(term_to_matrix(term), to_qubits(get_term_support(term)), N);
  }

  Eigen::MatrixXcd U = (-t.value() * H).exp();
  return std::make_shared<MatrixGate>(U, support);
}

std::string MajoranaGate::to_string() const {
  if (terms.size() == 0) {
    return "";
  }

  std::vector<std::string> s;
  for (const auto& term : terms) {
    s.push_back(fmt::format("{:.3f} a_{} a_{}", term.a, term.i, term.j));
  }

  return fmt::format("{}", fmt::join(s, " + "));
}

FreeFermionGate::FreeFermionGate(const MajoranaGate& gate) : num_qubits(gate.num_qubits), t(gate.t), adj(gate.adj) {
  for (const auto& term : gate.terms) {
    size_t i = term.i;
    size_t j = term.j;
    
    size_t n = i / 2;
    size_t m = j / 2;
    if (i % 2 == 0 && j % 2 == 0) {
      add_term(n, m, gates::i*term.a, true);
      add_term(n, m, gates::i*term.a, false);
    } else if (i % 2 == 0) {
      add_term(n, m, term.a, true);
      add_term(n, m,-term.a, false);
    } else if (j % 2 == 0) {
      add_term(n, m,-term.a, true);
      add_term(n, m,-term.a, false);
    } else {
      add_term(n, m, gates::i*term.a, true);
      add_term(n, m,-gates::i*term.a, false);
    }
  }
}

void FreeFermionGate::add_term(uint32_t i, uint32_t j, std::complex<double> a, bool adj) {
  uint32_t i1 = std::min(i, j);
  uint32_t i2 = std::max(i, j);
  double sign = (i1 != i) ? -1.0 : 1.0;
  std::complex<double> amplitude = sign * a;
  
  auto idx = std::make_tuple(i1, i2, adj);
  if (!term_map.contains(idx)) {
    term_map[idx] = terms.size();
    terms.push_back({i1, i2, 0.0, adj});
  }

  int k = term_map[idx];
  terms[k].a += amplitude;
}

FreeFermionGate FreeFermionGate::combine(const FreeFermionGate& other) const {
  if (!t || !other.t) {
    throw std::runtime_error("Cannot combine MajoranaGates with unbound t.");
  }

  FreeFermionGate gate(*this);
  for (auto& term : gate.terms) {
    term.a *= gate.t.value();
  }

  for (const auto& term : other.terms) {
    gate.add_term(term.i, term.j, other.t.value()*term.a, term.adj);
  }

  gate.set_t(1.0);
  return gate;
}

std::string FreeFermionGate::label() const {
  std::string suffix = t ? fmt::format("({:.3f})\n", t.value()) : "";
  return fmt::format("MG{}", suffix);
}

FreeFermionGate FreeFermionGate::bind_parameters(const std::vector<double>& params) const {
  FreeFermionGate gate(*this);
  if (params.size() == 0) {
    if (gate.t) { 
      return std::move(gate);
    } else {
      throw std::runtime_error("FreeFermionGate has unbound parameter, but was passed no parameters to bind.");
    }
  } else if (params.size() == 1) {
    if (gate.t) {
      throw std::runtime_error("FreeFermionGate has already bound parameter, but was passed a parameters to bind.");
    } else {
      gate.t = params[0];
      return std::move(gate);
    }
  } else {
    throw std::runtime_error("FreeFermionGate was passed an incorrect number of parameters to bind.");
  }
}

void FreeFermionGate::apply_qubit_map(const Qubits& qubits) {
  for (auto& term : terms) {
    term.i = qubits[term.i];
    term.j = qubits[term.j];
  }
}

Eigen::MatrixXcd FreeFermionGate::to_matrix() const {
  auto gate = to_gate();
  auto m = gate->define();
  return embed_unitary(gate->define(), gate->qubits, num_qubits);
}

Eigen::MatrixXcd term_to_matrix(const QuadraticFermionTerm& term) {
  Eigen::Matrix2cd sm = (gates::X::value - gates::i*gates::Y::value)/2.0;
  
  size_t N = std::abs(static_cast<int>(term.j) - static_cast<int>(term.i));
  std::vector<Eigen::Matrix2cd> p;
  for (size_t i = 0; i < N; i++) {
    p.push_back(gates::Z::value.asDiagonal());
  }
  if (term.adj) {
    p.push_back(sm.adjoint());
  } else {
    p.push_back(sm);
  }
  p[0] = sm * p[0];

  std::complex<double> a = term.a;
  if (term.j < term.i && !term.adj) {
    a = -a;
  }

  Eigen::MatrixXcd g = a * p[0];

  for (uint32_t i = 1; i < p.size(); i++) {
    Eigen::MatrixXcd gi = p[i];
    Eigen::MatrixXcd g0 = g;
    g = Eigen::kroneckerProduct(gi, g0);
  }

  return g + g.adjoint();
}

QubitInterval get_term_support(const QuadraticFermionTerm& term) {
  uint32_t i = std::min(term.i, term.j);
  uint32_t j = std::max(term.i, term.j);
  return std::make_pair(i, j+1);
}

Qubits FreeFermionGate::get_support() const {
  std::set<uint32_t> support;

  // Get total support
  for (const auto& term : terms) {
    Qubits term_support = to_qubits(get_term_support(term));
    for (auto q : term_support) {
      support.insert(q);
    }
  }

  return Qubits(support.begin(), support.end());
}

std::shared_ptr<Gate> FreeFermionGate::to_gate() const {
  if (!t) {
    throw std::runtime_error("Cannot convert a FreeFermionGate with unbound parameter to a matrix. Call bind_parameters([t]) first.");
  }

  Qubits support = get_support();
  FreeFermionGate gate(*this);
  Qubits map = reduced_support(support, num_qubits);
  gate.apply_qubit_map(map);

  size_t N = support.size();
  Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(1u << N, 1u << N);
  for (const auto& term : gate.terms) {
    H += embed_unitary(term_to_matrix(term), to_qubits(get_term_support(term)), N);
  }

  Eigen::MatrixXcd U = (gates::i * t.value() * H).exp();
  return std::make_shared<MatrixGate>(U, support);
}

Eigen::MatrixXcd FreeFermionGate::to_hamiltonian() const {
  Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(num_qubits, num_qubits);
  Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(num_qubits, num_qubits);

  for (const auto& term : terms) {
    if (term.adj) {
      A(term.i, term.j) += term.a;
      A(term.j, term.i) += term.a;
    } else {
      B(term.i, term.j) += term.a;
      B(term.j, term.i) += -term.a;
    }
  }

  Eigen::MatrixXcd Hm(2*num_qubits, 2*num_qubits);
  Hm << A,            B,
        B.adjoint(), -A.transpose();

  return Hm;
}

std::string FreeFermionGate::to_string() const {
  if (terms.size() == 0) {
    return "";
  }

  std::vector<std::string> s;
  for (const auto& term : terms) {
    s.push_back(fmt::format("({:.3f} + {:.3f}i) c_{}{} c_{}", std::real(term.a), std::imag(term.a), term.i, term.adj ? "^dag" : "", term.j));
  }

  return fmt::format("{} + h.c.", fmt::join(s, " + "));
}

void CommutingHamiltonianGate::add_term(double a, const PauliString& p, const Qubits& qubits) {
  if (p.num_qubits != qubits.size()) {
    throw std::runtime_error(fmt::format("Invalid number of qubits passed to CommutingHamiltonianGate.add_term()."));
  }

  std::unordered_map<uint32_t, uint32_t> pos1;
  for (size_t i = 0; i < qubits.size(); i++) {
    pos1[qubits[i]] = i;
  }

  // Check for non-commuting terms
  std::unordered_set<uint32_t> s(qubits.begin(), qubits.end());
  for (const auto& term : terms) {
    // Find overlapping qubits
    std::vector<uint32_t> overlap;
    for (uint32_t x : term.support) {
      if (s.contains(x)) {
        overlap.push_back(x);
      }
    }

    std::unordered_map<uint32_t, uint32_t> pos2;
    for (size_t i = 0; i < term.support.size(); i++) {
      pos2[term.support[i]] = i;
    }

    size_t commuting_indices = 0;
    for (uint32_t q : overlap) {
      uint32_t r1 = pos1[q];
      uint32_t r2 = pos2[q];

      bool x1 = p.get_x(r1);
      bool x2 = term.pauli.get_x(r2);
      bool z1 = p.get_z(r1);
      bool z2 = term.pauli.get_z(r2);

      if ((x1 == x2 && z1 == z2) || (!x1 && !z1) || (!x2 && !z2)) {
        commuting_indices++;
      }
    }

    size_t anticommuting_indices = overlap.size() - commuting_indices;

    if (anticommuting_indices % 2 != 0) {
      throw std::runtime_error(fmt::format("Operators do not commute: {} and {}\n", p, term.pauli));
    }
  }

  terms.push_back({a, p, qubits});
}

std::string CommutingHamiltonianGate::label() const {
  std::string suffix = t ? fmt::format("({:.3f})\n", t.value()) : "";
  return fmt::format("CH{}", suffix);
}

CommutingHamiltonianGate CommutingHamiltonianGate::bind_parameters(const std::vector<double>& params) const {
  CommutingHamiltonianGate gate(*this);
  if (params.size() == 0) {
    if (gate.t) { 
      return std::move(gate);
    } else {
      throw std::runtime_error("CommutingHamiltonianGate has unbound parameter, but was passed no parameters to bind.");
    }
  } else if (params.size() == 1) {
    if (gate.t) {
      throw std::runtime_error("CommutingHamiltonianGate has already bound parameter, but was passed a parameters to bind.");
    } else {
      gate.t = params[0];
      return std::move(gate);
    }
  } else {
    throw std::runtime_error("CommutingHamiltonianGate was passed an incorrect number of parameters to bind.");
  }
}

void CommutingHamiltonianGate::apply_qubit_map(const Qubits& qubits) {
  for (auto& [a, p, support] : terms) {
    for (int i = 0; i < support.size(); i++) {
      support[i] = qubits[support[i]];
    }
  }
}

Eigen::MatrixXcd CommutingHamiltonianGate::to_matrix() const {
  auto gate = to_gate();
  auto m = gate->define();
  return embed_unitary(gate->define(), gate->qubits, num_qubits);
}

Eigen::MatrixXcd term_to_matrix(const PauliTerm& term) {
  return term.a * term.pauli.to_matrix();
}

Qubits CommutingHamiltonianGate::get_support() const {
  std::set<uint32_t> support;

  // Get total support
  for (const auto& term : terms) {
    Qubits term_support = term.support;
    for (auto q : term_support) {
      support.insert(q);
    }
  }

  return Qubits(support.begin(), support.end());
}

std::shared_ptr<Gate> CommutingHamiltonianGate::to_gate() const {
  if (!t) {
    throw std::runtime_error("Cannot convert a CommutingHamiltonianGate with unbound parameter to a matrix. Call bind_parameters([t]) first.");
  }

  Qubits support = get_support();
  CommutingHamiltonianGate gate(*this);
  Qubits map = reduced_support(support, num_qubits);
  gate.apply_qubit_map(map);

  size_t N = support.size();
  Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(1u << N, 1u << N);
  for (const auto& term : gate.terms) {
    H += embed_unitary(term_to_matrix(term), term.support, N);
  }

  Eigen::MatrixXcd U = (gates::i * t.value() * H).exp();
  return std::make_shared<MatrixGate>(U, support);
}

std::string CommutingHamiltonianGate::to_string() const {
  if (terms.size() == 0) {
    return "";
  }

  std::vector<std::string> s;
  for (const auto& term : terms) {
    s.push_back(fmt::format("{:.3f}*{} ({})", term.a, term.pauli, term.support));
  }

  return fmt::format("{} {}", fmt::join(s, " + "), adj ? "(dag)" : "");
}

QuantumInstruction copy_quantum_instruction(const QuantumInstruction& qinst) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](std::shared_ptr<Gate> gate) {
      return QuantumInstruction(gate->clone());
    },
    [](const FreeFermionGate& gate) {
      return QuantumInstruction(gate);;
    },
    [](const CommutingHamiltonianGate& gate) {
      return QuantumInstruction(gate);;
    },
    [](const Measurement& m) {
      return QuantumInstruction(m);
    },
    [](const WeakMeasurement& m) {
      return QuantumInstruction(m);
    }
  }, qinst);
}

Instruction copy_instruction(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded {
      [](const QuantumInstruction& qinst) -> Instruction {
        return copy_quantum_instruction(qinst);
      },
      [](const ClassicalInstruction& clinst) -> Instruction {
        return ClassicalInstruction(clinst);
      },
      [](const ConditionedInstruction& cinst) -> Instruction {
        return ConditionedInstruction(copy_quantum_instruction(cinst.inst), cinst.control, cinst.target);
      }
    }, inst);
}

Measurement::Measurement(const Qubits& qubits, std::optional<PauliString> pauli, std::optional<bool> outcome)
: qubits(qubits), pauli(pauli), outcome(outcome) {
  PauliString p = pauli ? pauli.value() : PauliString("+Z");
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("Invalid number of qubits {} passed to measurement of pauli {}.", qubits, pauli.value()));
  }

  if (!p.hermitian()) {
    throw std::runtime_error(fmt::format("Cannot perform measurement on non-Hermitian Pauli string {}.", p));
  }

  if (qubits.size() == 0) {
    throw std::runtime_error("Must perform measurement on nonzero qubits.");
  }
}

Measurement Measurement::computational_basis(uint32_t q, std::optional<bool> outcome) {
  return Measurement({q}, PauliString("+Z"), outcome);
}

PauliString Measurement::get_pauli() const {
  if (pauli) {
    return pauli.value();
  } else {
    return PauliString("+Z");
  }
}

bool Measurement::is_basis() const {
  return !pauli || (pauli == PauliString("+Z"));
}

bool Measurement::is_forced() const {
  return bool(outcome);
}

bool Measurement::get_outcome() const {
  // is_forced() MUST be true, otherwise this will throw an exception
  return outcome.value();
}

WeakMeasurement::WeakMeasurement(const Qubits& qubits, std::optional<double> beta, std::optional<PauliString> pauli, std::optional<bool> outcome)
  : qubits(qubits), beta(beta), pauli(pauli), outcome(outcome) {
  PauliString p = pauli ? pauli.value() : PauliString("+Z");
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("Invalid number of qubits {} passed to weak measurement of pauli {}.", qubits, pauli.value()));
  }
}

size_t WeakMeasurement::num_params() const {
  return beta ? 0 : 1;
}

WeakMeasurement WeakMeasurement::bind_parameters(const std::vector<double>& params) const {
  WeakMeasurement m(*this);
  if (params.size() == 0) {
    if (m.beta) { 
      return std::move(m);
    } else {
      throw std::runtime_error("WeakMeasurement has unbound parameter, but was passed no parameters to bind.");
    }
  } else if (params.size() == 1) {
    if (m.beta) {
      throw std::runtime_error("WeakMeasurement has already bound parameter, but was passed a parameters to bind.");
    } else {
      m.beta = params[0];
      return std::move(m);
    }
  } else {
    throw std::runtime_error("WeakMeasurement was passed an incorrect number of parameters to bind.");
  }
}

PauliString WeakMeasurement::get_pauli() const {
  if (pauli) {
    return pauli.value();
  } else {
    return PauliString("+Z");
  }
}

bool WeakMeasurement::is_basis() const {
  return !pauli || (pauli == PauliString("+Z"));
}

bool WeakMeasurement::is_forced() const {
  return bool(outcome);
}

bool WeakMeasurement::get_outcome() const {
  // is_forced() MUST be true, otherwise this will throw an exception
  return outcome.value();
}

size_t get_quantum_instruction_num_params(const QuantumInstruction& qinst) {
	return std::visit(quantumcircuit_utils::overloaded {
    [](const std::shared_ptr<Gate> gate) -> size_t { 
      return gate->num_params();
    },
    [](const FreeFermionGate& gate) -> size_t {
      return gate.num_params();
    },
    [](const CommutingHamiltonianGate& gate) -> size_t {
      return gate.num_params();
    },
    [](const Measurement& m) -> size_t { 
      return 0;
    },
    [](const WeakMeasurement& m) -> size_t {
      return m.num_params();
    }
  }, qinst);
}

size_t get_instruction_num_params(const Instruction& inst) {
	return std::visit(quantumcircuit_utils::overloaded {
    [](const QuantumInstruction& qinst) -> size_t {
      return get_quantum_instruction_num_params(qinst);
    },
    [](const ClassicalInstruction& clinst) -> size_t {
      return 0;
    },
    [](const ConditionedInstruction& cinst) -> size_t { 
      return get_quantum_instruction_num_params(cinst.inst);
    },
  }, inst);
}

Qubits get_quantum_instruction_support(const QuantumInstruction& qinst) {
	return std::visit(quantumcircuit_utils::overloaded {
    [](const std::shared_ptr<Gate> gate) { 
      return gate->qubits;
    },
    [](const FreeFermionGate& gate) {
      return gate.get_support();
    },
    [](const CommutingHamiltonianGate& gate) {
      return gate.get_support();
    },
    [](const Measurement& m) { 
      return m.qubits;
    },
    [](const WeakMeasurement& m) {
      return m.qubits;
    }
  }, qinst);
}

Qubits get_instruction_support(const Instruction& inst) {
	return std::visit(quantumcircuit_utils::overloaded {
    [](const QuantumInstruction& qinst) -> Qubits {
      return get_quantum_instruction_support(qinst);
    },
    [](const ClassicalInstruction& clinst) -> Qubits {
      return {};
    },
    [](const ConditionedInstruction& cinst) -> Qubits { 
      return get_quantum_instruction_support(cinst.inst);
    },
  }, inst);
}

Qubits get_instruction_classical_support(const Instruction& inst) {
	return std::visit(quantumcircuit_utils::overloaded {
    [](const QuantumInstruction& qinst) -> Qubits {
      return {};
    },
    [](const ClassicalInstruction& clinst) -> Qubits {
      return clinst.bits;
    },
    [](const ConditionedInstruction& cinst) -> Qubits { 
      Qubits support;
      if (cinst.control) {
        support.push_back(cinst.control.value());
      }

      if (cinst.target) {
        support.push_back(cinst.target.value());
      }

      return support;
    },
  }, inst);
}

bool quantum_instruction_is_unitary(const QuantumInstruction& qinst) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](std::shared_ptr<Gate> gate) { return true; },
    [](const FreeFermionGate& gate) { return true; },
    [](const CommutingHamiltonianGate& gate) { return true; },
    [](const Measurement& m) { return false; },
    [](const WeakMeasurement& m) { return false; }
  }, qinst);
}

bool instruction_is_unitary(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded {
    [](const QuantumInstruction& qinst) { 
      return quantum_instruction_is_unitary(qinst);
    },
    [](const ClassicalInstruction& clinst) { 
      return true;
    }, 
    [](const ConditionedInstruction& cinst) { 
      return quantum_instruction_is_unitary(cinst.inst);
    }, 
  }, inst);
}

Instruction instruction_adjoint(const Instruction& inst) {
  auto qinst_adjoint = [](const QuantumInstruction& qinst) {
    return std::visit(quantumcircuit_utils::overloaded {
      [](std::shared_ptr<Gate> gate) -> QuantumInstruction {
        return gate->adjoint();
      },
      [](const FreeFermionGate& gate) -> QuantumInstruction {
        return gate.adjoint();
      },
      [](const CommutingHamiltonianGate& gate) -> QuantumInstruction {
        return gate.adjoint();
      },
      [](const Measurement& m) -> QuantumInstruction {
        throw std::runtime_error("Cannot return adjoint of non-unitary operations.");
      },
      [](const WeakMeasurement& m) -> QuantumInstruction {
        throw std::runtime_error("Cannot return adjoint of non-unitary operations.");
      }
    }, qinst);
  };

  return std::visit(quantumcircuit_utils::overloaded{ 
    [&](const QuantumInstruction& qinst) -> Instruction {
      return qinst_adjoint(qinst);
    },
    [](const ClassicalInstruction& clinst) -> Instruction {
      return clinst;
    },
    [&](const ConditionedInstruction& cinst) -> Instruction {
      return ConditionedInstruction(qinst_adjoint(cinst.inst), cinst.control, cinst.target);
    }
  }, inst);
}

bool instruction_is_classical(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded{ 
    [&](const QuantumInstruction& qinst) {
      return false;
    },
    [](const ClassicalInstruction& clinst) {
      return true;
    },
    [&](const ConditionedInstruction& cinst) {
      return cinst.control || cinst.target;
    }
  }, inst);
}

bool instruction_is_quantum(const Instruction& inst) {
  return std::visit(quantumcircuit_utils::overloaded{ 
    [&](const QuantumInstruction& qinst) {
      return true;
    },
    [](const ClassicalInstruction& clinst) {
      return false;
    },
    [&](const ConditionedInstruction& cinst) {
      return true;
    }
  }, inst);
}
