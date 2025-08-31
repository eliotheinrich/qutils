#include "GaussianState.h"

#include <sstream>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

void GaussianState::particles_at(const Qubits& sites) {
  for (auto i : sites) {
    if (i > L) {
      throw std::invalid_argument(fmt::format("Invalid site. Must be within 0 < i < {}", L));
    }
  }

  amplitudes = Eigen::MatrixXcd::Zero(2*L, L);
  std::vector<bool> included(L, false);
  for (auto i : sites) {
    included[i] = true;
  }

  for (size_t i = 0; i < L; i++) {
    if (included[i]) {
      amplitudes(i + L, i) = 1.0;
    } else {
      amplitudes(i, i) = 1.0;
    }
  }
}

void GaussianState::single_particle() {
  particles_at({L/2});
}

void GaussianState::all_particles() {
  Qubits sites(L);
  std::iota(sites.begin(), sites.end(), 0);
  particles_at(sites);
}

void GaussianState::checkerboard_particles() {
  Qubits sites;
  for (size_t i = 0; i < L; i++) {
    sites.push_back(i);
  }
  particles_at(sites);
}

GaussianState::GaussianState(uint32_t L, std::optional<Qubits> sites) : FreeFermionState(L) {
  if (sites) {
    particles_at(sites.value());
  } else {
    particles_at({});
  }
}

void GaussianState::swap(size_t i, size_t j) {
  amplitudes.row(i).swap(amplitudes.row(j));
  amplitudes.row(i + L).swap(amplitudes.row(j + L));
}

double GaussianState::entanglement(const QubitSupport& support, uint32_t index) {
  auto sites = to_qubits(support);
  size_t N = sites.size();
  if (N == 0) {
    return 0.0;
  }

  if (N > L/2) {
    auto _support = support_complement(support, L);
    return entanglement(_support, index);
  }

  auto C = covariance_matrix();

  std::vector<int> _sites(sites.begin(), sites.end());
  for (size_t i = 0; i < N; i++) {
    _sites.push_back(sites[i] + L);
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

bool GaussianState::is_identity(const Eigen::MatrixXcd& A) const {
  size_t r = A.rows();
  size_t c = A.cols();

  if (r != c) {
    throw std::runtime_error("Non-square matrix passed to is_identity.");
  }

  Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(r, c);

  return (A - I).norm() < 1e-4;
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

bool GaussianState::check_orthogonality() const {
  auto A = amplitudes.adjoint() * amplitudes;
  return is_identity(A);
}

void GaussianState::assert_ortho() const {
  bool ortho = check_orthogonality();
  if (!ortho) {
    throw std::runtime_error("Not orthogonal!");
  }
}

Eigen::MatrixXcd GaussianState::prepare_hamiltonian(const FreeFermionHamiltonian& H) const {
  Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(L, L);
  for (const auto& term : H.terms) {
    A(term.i, term.j) = term.a;
    if (term.i != term.j) {
      A(term.j, term.i) = term.a;
    }
  }

  Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(L, L);
  for (const auto& term : H.nc_terms) {
    B(term.i, term.j) = term.a;
    B(term.j, term.i) = -term.a;
  }

  Eigen::MatrixXcd Hm(2*L, 2*L);
  Hm << A,            B,
     B.adjoint(), -A.transpose();

  return Hm;
}

void GaussianState::evolve(const Eigen::MatrixXcd& U) {
  if (!is_unitary(U)) {
    throw std::runtime_error("Non-unitary matrix passed to evolve.");
  }

  amplitudes = U * amplitudes;
}

void GaussianState::evolve_hamiltonian(const FreeFermionHamiltonian& H, double t) {
  Eigen::MatrixXcd U = (std::complex<double>(0.0, -t) * prepare_hamiltonian(H)).exp();
  evolve(U);
}

void GaussianState::weak_measurement(const Eigen::MatrixXcd& U) {
  amplitudes = U * amplitudes;
  orthogonalize();
}

void GaussianState::weak_measurement_hamiltonian(const FreeFermionHamiltonian& H, double beta) {
  Eigen::MatrixXcd U = (std::complex<double>(beta, 0.0) * prepare_hamiltonian(H)).exp();
  weak_measurement(U);
}

void GaussianState::forced_projective_measurement(size_t i, bool outcome) {
  if (i < 0 || i > L) {
    throw std::invalid_argument(fmt::format("Invalid qubit measured: {}, L = {}", i, L));
  }

  size_t k = outcome ? (i + L) : i;
  size_t k_ = outcome ? i : (i + L);

  size_t i0;
  double d = 0.0;
  for (size_t j = 0; j < L; j++) {
    double dj = std::abs(amplitudes(k, j));
    if (dj > d) {
      d = dj;
      i0 = j;
    }
  }

  if (!(d > 0)) {
    std::cout << fmt::format("d = {:.5f}\n", d);
    throw std::runtime_error("Found no positive amplitudes to determine i0.");
  }

  for (size_t j = 0; j < L; j++) {
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
}

double GaussianState::num_particles() const {
  auto C = covariance_matrix();
  return C.trace().real();
}

double GaussianState::num_real_particles() const {
  auto C = covariance_matrix().block(L, L, L, L);
  return C.trace().real();
}

Eigen::MatrixXcd GaussianState::covariance_matrix() const {
  return amplitudes * amplitudes.adjoint();
}

double GaussianState::occupation(size_t i) const {
  double d = 0.0;

  for (size_t j = 0; j < L; j++) {
    auto c = std::abs(amplitudes(i + L, j));
    d += c*c;
  }

  return d;
}

std::vector<double> GaussianState::occupation() const {
  auto C = covariance_matrix();
  std::vector<double> n(L);

  for (size_t i = 0; i < L; i++) {
    n[i] = std::abs(C(i + L, i + L));
  }

  return n;
}

std::string GaussianState::to_string() const {
  std::stringstream s;
  s << amplitudes;
  return s.str();
}
