#include "GaussianState.h"

#include <sstream>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

void GaussianState::particles_at(const Qubits& sites) {
  for (auto i : sites) {
    if (i > num_qubits) {
      throw std::invalid_argument(fmt::format("Invalid site. Must be within 0 < i < {}", num_qubits));
    }
  }

  amplitudes = Eigen::MatrixXcd::Zero(2*num_qubits, num_qubits);
  std::vector<bool> included(num_qubits, false);
  for (auto i : sites) {
    included[i] = true;
  }

  for (size_t i = 0; i < num_qubits; i++) {
    if (included[i]) {
      amplitudes(i + num_qubits, i) = 1.0;
    } else {
      amplitudes(i, i) = 1.0;
    }
  }
}

void GaussianState::single_particle() {
  particles_at({num_qubits/2});
}

void GaussianState::all_particles() {
  Qubits sites(num_qubits);
  std::iota(sites.begin(), sites.end(), 0);
  particles_at(sites);
}

void GaussianState::checkerboard_particles() {
  Qubits sites;
  for (size_t i = 0; i < num_qubits; i++) {
    sites.push_back(i);
  }
  particles_at(sites);
}

GaussianState::GaussianState(uint32_t num_qubits, std::optional<Qubits> sites) : MagicQuantumState(num_qubits) {
  if (sites) {
    particles_at(sites.value());
  } else {
    particles_at({});
  }
}

void GaussianState::swap(size_t i, size_t j) {
  amplitudes.row(i).swap(amplitudes.row(j));
  amplitudes.row(i + num_qubits).swap(amplitudes.row(j + num_qubits));
}

double GaussianState::entanglement(const QubitSupport& support, uint32_t index) {
  auto sites = to_qubits(support);
  size_t N = sites.size();
  if (N == 0) {
    return 0.0;
  }

  if (N > num_qubits/2) {
    auto _support = support_complement(support, num_qubits);
    return entanglement(_support, index);
  }

  auto C = covariance_matrix();

  std::vector<int> _sites(sites.begin(), sites.end());
  for (size_t i = 0; i < N; i++) {
    _sites.push_back(sites[i] + num_qubits);
  }
  Eigen::VectorXi indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(_sites.data(), _sites.size());
  Eigen::MatrixXcd CA1 = C(indices, indices);
  Eigen::MatrixXcd CA2 = Eigen::MatrixXcd::Identity(CA1.rows(), CA1.cols()) - CA1;

  if (index == 1) {
    Eigen::MatrixXcd Cn = Eigen::MatrixXcd::Zero(indices.size(), indices.size());
    if (std::abs(CA1.determinant()) > 1e-6) {
      Cn += CA1*CA1.log();
    }

    if (std::abs(CA2.determinant()) > 1e-6) {
      Cn += CA2*CA2.log();
    }

    return -Cn.trace().real();
  } else {
    Eigen::MatrixXcd Cn = (CA1.pow(index) + CA2.pow(index)).log();
    return Cn.trace().real()/(2.0*static_cast<double>(1.0 - index));
  }
}

void GaussianState::orthogonalize() {
  size_t r = amplitudes.rows();
  size_t c = amplitudes.cols();
  Eigen::MatrixXcd Q(r, c);
  for (uint32_t i = 0; i < c; i++) {
    Eigen::VectorXcd q = amplitudes.col(i);

    for (int j = 0; j < i; ++j) {
      q -= Q.col(j).adjoint() * amplitudes.col(i) * Q.col(j);
    }

    q.normalize();
    Q.col(i) = q;
  }

  amplitudes = Q;
}

void GaussianState::evolve(const FreeFermionGate& gate) {
  if (!gate.t) {
    throw std::runtime_error("Must bind a value to FreeFermionGate.t before applying.");
  }

  Eigen::MatrixXcd H = gate.to_hamiltonian();
  Eigen::MatrixXcd U = (-gates::i * gate.t.value() * H).exp();

  if (!is_unitary(U)) {
    throw std::runtime_error("Non-unitary matrix passed to evolve.");
  }

  amplitudes = U * amplitudes;
}

MeasurementData GaussianState::wmzr(uint32_t q, double beta, std::optional<bool> outcome) {
  double pZ = 1 - 2*occupation(q);
  double prob_zero = (1 + std::tanh(2*beta) * (1. - 2.*occupation(q)))/2.0;

  bool b;
  if (outcome) { // forced 
    b = outcome.value(); 
  } else {
    b = randf() >= prob_zero;
  }

  if (b) {
    beta = -beta;
  }

  FreeFermionGate gate(num_qubits);
  gate.add_term(q, q, 1.0);

  Eigen::MatrixXcd H = gate.to_hamiltonian();
  Eigen::MatrixXcd U = (beta * H).exp();

  amplitudes = U * amplitudes;
  orthogonalize();

  double prob_outcome = b ? (1.0 - prob_zero) : prob_zero;
  return {b, prob_outcome};
}

MeasurementData GaussianState::weak_measure(const WeakMeasurement& m) {
  if (m.num_params() > 0) {
    throw std::runtime_error("Cannot apply weak measurement with unbound strength.");
  }

  if (!m.is_basis()) {
    throw not_implemented();
  }

  return wmzr(m.qubits[0], m.beta.value(), m.outcome);
}

//void GaussianState::weak_measurement_hamiltonian(const FreeFermionGate& gate, double beta) {
//  Eigen::MatrixXcd U = (std::complex<double>(beta, 0.0) * gate.to_hamiltonian()).exp();
//  weak_measurement(U);
//}

MeasurementData GaussianState::mzr(uint32_t i, std::optional<bool> outcome_opt) {
  double p = occupation(i);
  bool outcome = outcome_opt ? outcome_opt.value() : (randf() < p);

  if (i < 0 || i > num_qubits) {
    throw std::invalid_argument(fmt::format("Invalid qubit measured: {}, num_qubits = {}", i, num_qubits));
  }

  size_t k = outcome ? (i + num_qubits) : i;
  size_t k_ = outcome ? i : (i + num_qubits);

  size_t i0;
  double d = 0.0;
  for (size_t j = 0; j < num_qubits; j++) {
    double dj = std::abs(amplitudes(k, j));
    if (dj > d) {
      d = dj;
      i0 = j;
    }
  }

  if (!(d > 0)) {
    throw std::runtime_error("Found no positive amplitudes to determine i0.");
  }

  for (size_t j = 0; j < num_qubits; j++) {
    if (j == i0) {
      continue;
    }

    amplitudes(Eigen::indexing::all, j) = amplitudes(Eigen::indexing::all, j) - amplitudes(k, j)/amplitudes(k, i0) * amplitudes(Eigen::indexing::all, i0);
    amplitudes(k_, j) = 0.0;
  }

  for (size_t j = 0; j < amplitudes.rows(); j++) {
    amplitudes(j, i0) = 0.0;
  }

  amplitudes(k, i0) = 1.0;

  orthogonalize();

  double prob_outcome = outcome ? p : (1.0 - p);
  return {outcome, prob_outcome};
}

MeasurementData GaussianState::measure(const Measurement& m) {
  if (!m.is_basis()) {
    throw not_implemented();
  } 

  return mzr(m.qubits[0], m.outcome);
}

double GaussianState::num_particles() const {
  auto C = covariance_matrix();
  return C.trace().real();
}

double GaussianState::num_real_particles() const {
  auto C = covariance_matrix().block(num_qubits, num_qubits, num_qubits, num_qubits);
  return C.trace().real();
}

Eigen::MatrixXcd GaussianState::covariance_matrix() const {
  return amplitudes * amplitudes.adjoint();
}

Eigen::MatrixXcd GaussianState::majorana_covariance_matrix() const {
  Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(2*num_qubits, 2*num_qubits);
  for (size_t i = 0; i < num_qubits; i++) {
    U(2*i,   i)              = 1.0;
    U(2*i,   i + num_qubits) = 1.0;
    U(2*i+1, i)              = gates::i;
    U(2*i+1, i + num_qubits) =-gates::i;
  }

  Eigen::MatrixXcd C = covariance_matrix();
  return gates::i*(U * C * U.adjoint() - Eigen::MatrixXcd::Identity(2*num_qubits, 2*num_qubits));
}

std::complex<double> GaussianState::majorana_expectation(const std::vector<uint32_t>& indices) const {
  if (indices.size() % 2) {
    throw std::runtime_error("Can only evaluate an even number of Majorana operators using Wick's theorem.");
  }

  size_t p = indices.size() / 2;

  Eigen::MatrixXcd M = majorana_covariance_matrix();
  Eigen::MatrixXcd Mi = M(indices, indices);

  return std::pow(gates::i, -p) * std::abs(std::sqrt(Mi.determinant()));
}

double GaussianState::occupation(size_t i) const {
  double d = 0.0;

  for (size_t j = 0; j < num_qubits; j++) {
    auto c = std::abs(amplitudes(i + num_qubits, j));
    d += c*c;
  }

  return d;
}

std::vector<double> GaussianState::occupation() const {
  auto C = covariance_matrix();
  std::vector<double> n(num_qubits);

  for (size_t i = 0; i < num_qubits; i++) {
    n[i] = std::abs(C(i + num_qubits, i + num_qubits));
  }

  return n;
}

std::string GaussianState::to_string() const {
  std::stringstream s;
  s << amplitudes;
  return s.str();
}

std::complex<double> GaussianState::expectation(const PauliString& pauli) const {
  if (pauli.num_qubits != num_qubits) {
    throw std::runtime_error("Mismatched number of qubits in GaussianState.expectation(PauliString)");
  }

  if (pauli.is_basis()) {
    uint32_t q = pauli.get_support()[0];
    return 1. - 2.*occupation(q);
  }

  // Constructing the corresponding Majorana string
  int phase = 0;
  std::vector<uint32_t> parities(2*num_qubits);;
  for (size_t i = 0; i < num_qubits; i++) {
    if (pauli.get_z(i) && pauli.get_x(i)) { // Y
      phase = (phase + 3*i) % 4;
      for (size_t k = 0; k < i; k++) {
        parities[2*k]++;
        parities[2*k+1]++;
      }
      parities[2*i+1]++;
    } else if (pauli.get_x(i)) { // X
      phase = (phase + 3*i) % 4;
      for (size_t k = 0; k < i; k++) {
        parities[2*k]++;
        parities[2*k+1]++;
      }
      parities[2*i]++;
    } else if (pauli.get_z(i)) { // Z
      phase = (phase + 3) % 4;
      parities[2*i]++;
      parities[2*i+1]++;
    }
  }

  // Every Majorana that appears an odd number of times is included in the string
  std::vector<uint32_t> indices;
  for (size_t i = 0; i < parities.size(); i++) {
    if (parities[i] % 2) {
      indices.push_back(i);
    }
  }

  if (indices.size() % 2) {
    return 0.0;
  } else {
    // TODO check on this sign
    return std::pow(gates::i, phase) * majorana_expectation(indices);
  }
}

double GaussianState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  Qubits qubits;
  if (support) {
    qubits = to_qubits(support.value());
  } else {
    qubits = Qubits(num_qubits);
    std::iota(qubits.begin(), qubits.end(), 0);
  }

  // Construct a copy of the state and iteratively mzr it to get marginal distributions
  double d = 1.0;
  GaussianState tmp(*this);

  size_t i = 0;
  for (uint32_t q : qubits) {
    bool b = bits.get(i++);
    double p = tmp.occupation(q);
    double prob_outcome = b ? p : (1.0 - p);
    if (prob_outcome < 1e-6) {
      return 0.0;
    } else {
      tmp.mzr(q, b);
      d *= prob_outcome;
    }
  }

  return d;
}

std::vector<double> GaussianState::probabilities() const {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot compute fill probability distribution for n > 15 qubits.");
  }

  // Naive
  std::vector<double> probs(basis);
  for (uint32_t z = 0; z < basis; z++) {
    BitString bits = BitString::from_bits(num_qubits, z);
    probs[z] = expectation(bits);
  }

  return probs;
}

#include <glaze/glaze.hpp>

namespace glz::detail {
   template <>
   struct from<BEVE, Eigen::MatrixXcd> {
      template <auto Opts>
      static void op(Eigen::MatrixXcd& value, auto&&... args) {
        std::string str;
        read<BEVE>::op<Opts>(str, args...);
        std::vector<char> buffer(str.begin(), str.end());

        const size_t header_size = 2 * sizeof(size_t);
        size_t rows, cols;
        memcpy(&rows, buffer.data(), sizeof(size_t));
        memcpy(&cols, buffer.data() + sizeof(size_t), sizeof(size_t));
        
        Eigen::MatrixXcd matrix(rows, cols);
        memcpy(matrix.data(), buffer.data() + header_size, rows * cols * sizeof(std::complex<double>));

        value = matrix; 
      }
   };

   template <>
   struct to<BEVE, Eigen::MatrixXcd> {
      template <auto Opts>
      static void op(const Eigen::MatrixXcd& value, auto&&... args) noexcept {
        const size_t header_size = 2 * sizeof(size_t);
        const size_t rows = value.rows();
        const size_t cols = value.cols();
    
        const size_t data_size = rows * cols * sizeof(std::complex<double>);
        std::vector<char> buffer(header_size + data_size);
    
        memcpy(buffer.data(), &rows, sizeof(size_t));
        memcpy(buffer.data() + sizeof(size_t), &cols, sizeof(size_t));
        memcpy(buffer.data() + header_size, value.data(), data_size);

        std::string data(buffer.begin(), buffer.end());
        write<BEVE>::op<Opts>(data, args...);
      }
   };
}

struct GaussianState::glaze {
  using T = GaussianState;
  static constexpr auto value = glz::object(
    &T::amplitudes,
    &T::use_parent,
    &T::num_qubits,
    &T::basis
  );
};

DEFINE_SERIALIZATION(GaussianState);
