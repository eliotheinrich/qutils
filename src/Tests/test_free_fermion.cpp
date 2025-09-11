#include "tests.hpp"

#include <unsupported/Eigen/MatrixFunctions>
#include "FreeFermion.h"

#define MPS_DEBUG_LEVEL 1

template <typename T, typename V>
bool states_close(const T& first, const V& second) {
  DensityMatrix d1(first);
  DensityMatrix d2(second);

  if (d1.get_num_qubits() != d2.get_num_qubits()) {
    return false;
  }

  return (d1.data - d2.data).cwiseAbs().maxCoeff() < 1e-2;
}

template <typename T, typename V, typename... Args>
bool states_close(const T& first, const V& second, const Args&... args) {
  if (!states_close(first, second)) {
    return false;
  } else {
    return states_close(first, args...);
  }
}

template <typename T, typename... QuantumStates>
size_t get_num_qubits(const T& first, const QuantumStates&... args) {
  size_t num_qubits = first.get_num_qubits();

  if constexpr (sizeof...(args) == 0) {
    return num_qubits;
  } else {
    if (num_qubits != get_num_qubits(args...)) {
      throw std::runtime_error("Error; inappropriate states passed to get_num_qubits. Number of qubits do not match.");
    }

    return num_qubits;
  }
}


Qubits random_qubits(size_t num_qubits, size_t k) {
  std::minstd_rand rng(randi());
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  std::shuffle(qubits.begin(), qubits.end(), rng);

  Qubits r(qubits.begin(), qubits.begin() + k);
  return r;
}

Qubits random_boundary_qubits(size_t num_qubits, size_t k) {
  Qubits qubits;
  if (k == 0) {
    return qubits;
  }

  size_t k1 = randi(0, k+1);
  size_t k2 = k - k1;

  for (uint32_t q = 0; q < k1; q++) {
    qubits.push_back(q);
  }

  for (uint32_t q = 0; q < k2; q++) {
    qubits.push_back(num_qubits - q - 1);
  }
  
  return qubits;
}

QubitInterval random_interval(size_t num_qubits, size_t k) {
  uint32_t q1 = randi() % (num_qubits - k + 1);
  uint32_t q2 = q1 + k;

  return std::make_pair(q1, q2);
}


// Functions for generating matchgates
FreeFermionGate random_free_fermion_gate(size_t nqb, size_t num_terms, size_t max_dist=1, double min_a=-1.0, double max_a=1.0) {
  FreeFermionGate gate(nqb, 1.0);
  for (size_t i = 0; i < num_terms; i++) {
    uint32_t s = randi(0, max_dist+1);
    uint32_t q = randi(0, nqb - s - 1);
    double theta = randf(min_a, max_a);
    gate.add_term(q, q+s, theta, randi() % 2);
  }

  return gate;
}

auto T(double theta, uint32_t q, size_t nqb) {
  double sin = std::sin(theta);
  double cos = std::cos(theta);
  Eigen::Matrix4cd Tm;
  Tm << 1.0, 0.0,           0.0,           0.0,
        0.0, cos,           gates::i*sin, 0.0,
        0.0, gates::i*sin, cos,           0.0,
        0.0, 0.0,           0.0,           1.0;

  FreeFermionGate gate(nqb, 1.0);
  gate.add_term(q+1, q, theta);
  return gate;
}


auto G(double theta, uint32_t q, size_t nqb) {
  double sin = std::sin(theta);
  double cos = std::cos(theta);
  FreeFermionGate gate(nqb, 1.0);
  gate.add_term(q+1, q, theta, false);
  return gate;
}

auto R(double theta, uint32_t q, size_t nqb) {
  Eigen::Matrix2cd Rm;
  Rm << 1.0, 0.0,
        0.0, std::exp(gates::i*theta);
  FreeFermionGate gate(nqb, 1.0);
  gate.add_term(q, q, theta);
  return gate;
};

void normalize(std::vector<double>& p) {
  double N = 0.0;
  for (size_t i = 0; i < p.size(); i++) {
    N += p[i];
  }

  for (size_t i = 0; i < p.size(); i++) {
    p[i] /= N;
  }
}

bool test_add_term() {
  constexpr size_t nqb = 1;
  FreeFermionGate gate(nqb);
  gate.add_term(0, 0, 1.0);
  gate.add_term(0, 0, 1.0);

  return true;
}

bool test_term_to_matrix() {
  auto majorana_term1 = [](const QuadraticTerm& term, size_t nqb) {
    Eigen::MatrixXcd cd1 = fermion_operator(term.i, nqb);
    Eigen::MatrixXcd c2  = fermion_operator(term.j, nqb).adjoint();

    Eigen::MatrixXcd h1 = term.a * cd1 * c2;
    Eigen::MatrixXcd h2 = h1.adjoint();
    Eigen::MatrixXcd h = h1 + h2;
    return h;
  };

  auto majorana_term2 = [](const QuadraticTerm& term, size_t nqb) {
    Eigen::MatrixXcd cd1 = fermion_operator(term.i, nqb);
    Eigen::MatrixXcd cd2 = fermion_operator(term.j, nqb);

    Eigen::MatrixXcd h1 = term.a * cd1 * cd2;
    Eigen::MatrixXcd h2 = h1.adjoint();
    Eigen::MatrixXcd h = h1 + h2;
    return h;
  };

  // Single-qubit
  for (size_t i = 0; i < 100; i++) {
    double theta = randf(0, 2 * M_PI);
    QuadraticTerm term;
    Eigen::Matrix2cd H1;
    Eigen::Matrix2cd H2;
    Eigen::MatrixXcd H3;

    term = {0, 0, theta/2.0, true}; 
    H1 = term_to_matrix(term);
    H2 = Eigen::Matrix2cd::Zero(); H2(1, 1) = theta;
    H3 = majorana_term1(term, 1);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    term = {0, 0, theta/2.0, false}; 
    H1 = term_to_matrix(term);
    H2 = Eigen::Matrix2cd::Zero();
    H3 = majorana_term2(term, 1);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));
  }

  // Two-qubit
  for (size_t i = 0; i < 100; i++) {
    double theta = randf(0, 2 * M_PI);
    QuadraticTerm term;
    Eigen::Matrix4cd H1;
    Eigen::Matrix4cd H2;
    Eigen::Matrix4cd H3;

    term = {0, 1, theta, true}; 
    H1 = term_to_matrix(term);
    H2 = Eigen::Matrix4cd::Zero(); H2(2, 1) = theta; H2(1, 2) = theta;
    H3 = majorana_term1(term, 2);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    // Flipping sites does not affect term
    term = {1, 0, theta, true}; 
    H1 = term_to_matrix(term);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    term = {0, 1, theta, false}; 
    H1 = term_to_matrix(term);
    H2 = Eigen::Matrix4cd::Zero(); H2(0, 3) = theta; H2(3, 0) = theta;
    H3 = majorana_term2(term, 2);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    // Sign should be flipped on particle-nonconserving terms when sites are flipped
    term = {1, 0, theta, false}; 
    H1 = term_to_matrix(term);
    H2 = Eigen::Matrix4cd::Zero(); H2(0, 3) = -theta; H2(3, 0) = -theta;
    H3 = majorana_term2(term, 2);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));
  }

  // Three-qubit
  for (size_t i = 0; i < 100; i++) {
    double theta = randf(0, 2 * M_PI);
    QuadraticTerm term;
    Eigen::MatrixXcd H1;
    Eigen::MatrixXcd H2;
    Eigen::MatrixXcd H3;

    term = {0, 2, theta, true};
    H1 = term_to_matrix(term);
    H2 = Eigen::MatrixXcd::Zero(8, 8); H2(4, 1) = theta; H2(1, 4) = theta; H2(6, 3) = -theta; H2(3, 6) = -theta;
    H3 = majorana_term1(term, 3);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    // Flipping sites does not affect term
    term = {2, 0, theta, true};
    H1 = term_to_matrix(term);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    term = {0, 2, theta, false};
    H1 = term_to_matrix(term);
    H2 = Eigen::MatrixXcd::Zero(8, 8); H2(5, 0) = theta; H2(0, 5) = theta; H2(7, 2) = -theta; H2(2, 7) = -theta;
    H3 = majorana_term2(term, 3);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));

    // Sign should be flipped on particle-nonconserving terms when sites are flipped
    term = {2, 0, theta, false};
    H1 = term_to_matrix(term);
    H2 = Eigen::MatrixXcd::Zero(8, 8); H2(5, 0) = -theta; H2(0, 5) = -theta; H2(7, 2) = theta; H2(2, 7) = theta;
    H3 = majorana_term2(term, 3);
    ASSERT(H1.isApprox(H2) && H1.isApprox(H3));
  }


  return true;
}

bool test_hamiltonian_support() {
  constexpr size_t nqb = 10;

  FreeFermionGate H(nqb);

  H.add_term(0, 1, 1.0);
  H.add_term(3, 4, 1.0);
  H.add_term(4, 5, 1.0);

  FreeFermionGate gate(H);

  Qubits expected = {0, 1, 3, 4, 5};
  ASSERT(gate.get_support() == expected);

  return true;
}

bool test_to_gate() {
  constexpr size_t nqb = 6;

  for (size_t i = 0; i < 10; i++) {
    auto ff_gate = random_free_fermion_gate(nqb, 10, 1);

    QuantumCircuit qc1(nqb);
    qc1.add_gate(ff_gate);

    QuantumCircuit qc2(nqb);
    qc2.add_gate(ff_gate.to_gate());
    auto U1 = qc1.to_matrix();
    auto U2 = qc2.to_matrix();
    
    ASSERT(U1.isApprox(U2));
  }

  return true;
}

bool test_free_fermion_state() {
  constexpr size_t nqb = 8;

  Qubits sites = random_qubits(nqb, nqb/2);
  Statevector psi(nqb);
  for (auto q : sites) {
    psi.x(q);
  }
  GaussianState fermion_state(nqb, sites);

  std::vector<double> op_dist = {0.3, 0.3, 0.3, 0.1, 0.1};
  normalize(op_dist);
  std::mt19937 gen(randi());
  std::discrete_distribution<> dist(op_dist.begin(), op_dist.end());

  for (size_t k = 0; k < 20; k++) {
    int gate_type = dist(gen);
    double theta = randf(0, 2 * M_PI);

    if (gate_type == 0) {
      uint32_t q = randi(0, nqb);
      FreeFermionGate gate = R(theta, q, nqb);
      psi.evolve(gate);
      fermion_state.evolve(gate);
    } else if (gate_type == 1) {
      uint32_t q = randi(0, nqb - 1);
      FreeFermionGate gate = T(theta, q, nqb);
      psi.evolve(gate);
      fermion_state.evolve(gate);
    } else if (gate_type == 2) {
      uint32_t q = randi(0, nqb - 1);
      FreeFermionGate gate = G(theta, q, nqb);
      psi.evolve(gate);
      fermion_state.evolve(gate);
    } else if (gate_type == 3) {
      FreeFermionGate gate = random_free_fermion_gate(nqb, 4);
      psi.evolve(gate);
      fermion_state.evolve(gate);
    } else {
      uint32_t q = randi(0, nqb);
      auto [outcome, p] = psi.mzr(q);
      fermion_state.mzr(q, outcome);
    }

    // Check state equality
    std::vector<double> c1;
    std::vector<double> c2;
    for (uint32_t i = 0; i < nqb; i++) {
      PauliString Z(nqb);
      Z.set_z(i, 1);
      c1.push_back((1.0 - psi.expectation(Z).real()) / 2.0);
      c2.push_back(fermion_state.occupation(i));
    }

    uint32_t index = randi(1, 4);
    std::vector<double> s1 = psi.get_entanglement(index);
    std::vector<double> s2 = fermion_state.get_entanglement(index);

    for (size_t i = 0; i < nqb; i++) {
      ASSERT(is_close_eps(1e-4, s1[i], s2[i]), fmt::format("Entanglement {} at {} is not equal: \n{::.5f}\n{::.5f}", index, i, s1, s2));
      ASSERT(is_close_eps(1e-4, c1[i], c2[i]), fmt::format("Occupations at {} are not equal: \n{::.5f}\n{::.5f}", i, c1, c2));
    }
  }

  return true;
}


bool test_amplitudes() {
  constexpr size_t nqb = 8;

  for (size_t k = 0; k < 10; k++) {
    Qubits sites = random_qubits(nqb, nqb/2);
    Statevector psi(nqb);
    for (auto q : sites) {
      psi.x(q);
    }
    GaussianState fermion_state(nqb, sites);

    FreeFermionGate gate = random_free_fermion_gate(nqb, 4);
    fermion_state.evolve(gate);

    Eigen::MatrixXd M = fermion_state.majorana_covariance_matrix();

    std::vector<double> c1;
    std::vector<double> c2;
    for (size_t i = 0; i < nqb; i++) {
      c1.push_back(fermion_state.occupation(i));
      c2.push_back((1.0 - std::real(M(2*i, 2*i + 1))) / 2.0);
    }

    for (size_t i = 0; i < nqb; i++) {
      ASSERT(is_close_eps(1e-4, c1[i], c2[i]), fmt::format("Occupations at {} are not equal: \n{::.5f}\n{::.5f}", i, c1, c2));
    }
  }

  return true;
};

bool test_majorana_expectation() {
  constexpr size_t nqb = 8;

  for (size_t i = 0; i < 10; i++) {
    Statevector psi(nqb);
    GaussianState fermion_state(nqb);

    QuantumCircuit qc(nqb);
    for (size_t j = 0; j < 10; j++) {
      auto gate = random_free_fermion_gate(nqb, 2);
      qc.add_gate(gate);
    }

    psi.evolve(qc);
    fermion_state.evolve(qc);

    size_t p = randi(1, nqb);
    std::vector<uint32_t> idxs = random_qubits(2*nqb, 2*p);

    PauliString P(nqb);
    for (auto i : idxs) {
      P = P * majorana_operator(i, nqb);
    }

    double c1 = std::abs(fermion_state.majorana_expectation(idxs));
    double c2 = std::abs(psi.expectation(P));

    ASSERT(is_close_eps(1e-4, c1, c2));

    size_t index = randi(1, 4);
    std::vector<double> s1 = psi.get_entanglement(index);
    std::vector<double> s2 = fermion_state.get_entanglement(index);

    std::vector<double> c1v;
    std::vector<double> c2v;
    for (uint32_t i = 0; i < nqb; i++) {
      PauliString Z(nqb);
      Z.set_z(i, 1);
      c1v.push_back((1.0 - psi.expectation(Z).real()) / 2.0);
      c2v.push_back(fermion_state.occupation(i));
    }

    for (size_t i = 0; i < nqb; i++) {
      ASSERT(is_close_eps(1e-4, s1[i], s2[i]), fmt::format("Entanglement {} at {} is not equal: \n{::.5f}\n{::.5f}", index, i, s1, s2));
      ASSERT(is_close_eps(1e-4, c1v[i], c2v[i]), fmt::format("Occupations at {} are not equal: \n{::.5f}\n{::.5f}", i, c1v, c2v));
    }
  }

  return true;
}

bool test_paulistring_expectation() {
  constexpr size_t nqb = 8;

  for (size_t i = 0; i < 10; i++) {
    Statevector psi(nqb);
    GaussianState fermion_state(nqb);

    QuantumCircuit qc(nqb);
    for (size_t j = 0; j < 10; j++) {
      auto gate = random_free_fermion_gate(nqb, 2);
      qc.add_gate(gate);
    }

    psi.evolve(qc);
    fermion_state.evolve(qc);

    PauliString P = PauliString::randh(nqb);

    double c1 = std::abs(fermion_state.expectation(P));
    double c2 = std::abs(psi.expectation(P));

    ASSERT(is_close_eps(1e-4, c1, c2));
  }

  return true;
}

bool test_bitstring_expectation() {
  constexpr size_t nqb = 8;

  for (size_t i = 0; i < 10; i++) {
    Statevector psi(nqb);
    GaussianState fermion_state(nqb);

    QuantumCircuit qc(nqb);
    for (size_t j = 0; j < 10; j++) {
      auto gate = random_free_fermion_gate(nqb, 4);
      qc.add_gate(gate);
    }

    psi.evolve(qc);
    fermion_state.evolve(qc);

    for (size_t k = 0; k < 20; k++) {
      size_t p = randi(1, nqb);
      BitString bits = BitString::random(p);
      Qubits support = random_qubits(nqb, p);

      double c1 = std::abs(fermion_state.expectation(bits, support));
      double c2 = std::abs(psi.expectation(bits, support));

      ASSERT(is_close_eps(1e-4, c1, c2));
    }
  }

  return true;
}

bool test_mzr() {
  constexpr size_t nqb = 8;

  Statevector psi(nqb);
  GaussianState fermion_state(nqb);

  QuantumCircuit qc(nqb);
  for (size_t j = 0; j < 10; j++) {
    auto gate = random_free_fermion_gate(nqb, 4);
    qc.add_gate(gate);
  }

  psi.evolve(qc);
  fermion_state.evolve(qc);

  for (size_t i = 0; i < nqb; i++) {
    uint32_t q = randi(0, nqb);
    if (randi() % 2) {
      auto [outcome1, p1] = psi.mzr(q);
      auto [outcome2, p2] = fermion_state.mzr(q, outcome1);

      ASSERT(outcome1 == outcome2 && is_close(p1, p2));
    } else {
      auto [outcome1, p1] = fermion_state.mzr(q);
      auto [outcome2, p2] = psi.mzr(q, outcome1);

      ASSERT(outcome1 == outcome2 && is_close(p1, p2));
    }
  }

  return true;
}

bool test_weak_mzr() {
  constexpr size_t nqb = 8;

  Statevector psi(nqb);
  GaussianState fermion_state(nqb);

  QuantumCircuit qc(nqb);
  for (size_t j = 0; j < 10; j++) {
    auto gate = random_free_fermion_gate(nqb, 4);
    qc.add_gate(gate);
  }

  psi.evolve(qc);
  fermion_state.evolve(qc);

  double beta = 0.5;
  for (size_t i = 0; i < nqb; i++) {
    uint32_t q = randi(0, nqb);
    if (randi() % 2) {
      auto [outcome1, p1] = psi.wmzr(q, beta);
      auto [outcome2, p2] = fermion_state.wmzr(q, beta, outcome1);

      ASSERT(outcome1 == outcome2 && is_close(p1, p2));
    } else {
      auto [outcome1, p1] = fermion_state.wmzr(q, beta);
      auto [outcome2, p2] = psi.wmzr(q, beta, outcome1);

      ASSERT(outcome1 == outcome2 && is_close(p1, p2));
    }
  }

  return true;
}

bool test_majorana_to_pauli() {
  constexpr size_t nqb = 10;

  for (size_t i = 0; i < 100; i++) {
    PauliString pauli = PauliString::randh(nqb);

    auto majorana = pauli_to_majorana(pauli);

    PauliString pauli_ = majorana_to_pauli(majorana, nqb);

    ASSERT(pauli == pauli_);
  }

  return true;
}

bool test_majorana_covariance_matrix() {
  constexpr size_t nqb = 6;
  Statevector psi(nqb);
  GaussianState fermion_state(nqb);

  QuantumCircuit qc(nqb);
  for (size_t j = 0; j < 10; j++) {
    auto gate = random_free_fermion_gate(nqb, 4);
    qc.add_gate(gate);
  }

  psi.evolve(qc);
  fermion_state.evolve(qc);

  Eigen::MatrixXcd M = fermion_state.majorana_covariance_matrix();

  for (size_t m = 0; m < 2*nqb; m++) {
    for (size_t n = 0; n < 2*nqb; n++) {
      std::complex<double> c1 = M(m, n);
      PauliString g1 = majorana_operator(m, nqb);
      PauliString g2 = majorana_operator(n, nqb);
      std::complex<double> c2 = -gates::i /2.0 * (psi.expectation(g1 * g2) - psi.expectation(g2 * g1));
      ASSERT(is_close(c1, c2));
    }
  }

  return true;
}


bool test_sample_paulis() {
  constexpr size_t nqb = 3;

  int s = randi();
  Random::seed_rng(s);
  std::srand(s);
  std::cout << fmt::format("s = {}\n", s);

  QuantumCircuit qc(nqb);
  qc.add_gate(random_free_fermion_gate(nqb, 3*nqb, 1));

  GaussianState state(nqb);
  state.evolve(qc);

  MatrixProductState mps(nqb, 1u << nqb);
  mps.evolve(qc);

  size_t nsamples = 1;

  s = randi();
  Random::seed_rng(s);
  auto paulis1 = state.sample_paulis({}, nsamples);
  std::cout << "\n\n";
  Random::seed_rng(s);
  //auto paulis1_ = state.sample_paulis_old({}, nsamples);

  //for (size_t i = 0; i < nsamples; i++) {
  //  auto [p1, a1] = paulis1[i];
  //  auto [p2, a2] = paulis1_[i];
  //  std::cout << fmt::format("{}, {} -> {}, {}\n", p1, p2, a1[0], a2[0]);
  //  ASSERT(p1 == p2 && is_close(a1[0], a2[0]));
  //}

  auto paulis2 = mps.sample_paulis({}, nsamples);

  for (auto [p, a] : paulis1) {
    double c = std::abs(mps.expectation(p));
    std::cout << fmt::format("<{}> = {:.5f}, {:.5f}\n", p, a[0], c);
    ASSERT(is_close_eps(1e-5, a[0], c), fmt::format("<{}> = {:.5f}, {:.5f}\n", p, a[0], c));
  }

  return true;
}

//bool test_det() {
//  size_t n = 10;
//  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
//
//  // Initial computations
//  Eigen::MatrixXd inv_A = A.inverse();
//  double det = A.determinant();
//
//  for (size_t k = 0; k < 8; k++) {
//    if (randf() < 0.5) {
//      size_t q = randi(0, A.cols());
//      update_inv_and_det_lr(A, inv_A, det, q);
//    } else {
//      size_t q = randi(0, A.cols());
//      update_inv_and_det_resize(A, inv_A, det, q);
//    }
//
//    double det_tmp = A.determinant();
//    ASSERT(is_close(det, det_tmp));
//    std::cout << fmt::format("dets = {:.6f}, {:.6f}\n\n\n", det, det_tmp);
//  }
//
//  return true;
//}
//
//bool test_det_case() {
//  size_t n = 5;
//  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
//  std::cout << A << "\n";
//
//  Eigen::MatrixXd inv_A = A.inverse();
//  double det = A.determinant();
//
//  std::vector<size_t> inds = {0, 3};
//  std::vector<size_t> inds_;
//  for (size_t i = 0; i < n; i++) {
//    if (std::find(inds.begin(), inds.end(), i) == inds.end()) {
//      inds_.push_back(i);
//    }
//  }
//
//  std::cout << fmt::format("Before modification, det = {}\n", det);
//  update_inv_and_det_resize(A, inv_A, det, inds[0]);
//
//  double det_tmp = A.determinant();
//  std::cout << fmt::format("det = {}, {}\n", det, det_tmp);
//
//  std::cout << inv_A - A.inverse() << "\n";
//
//  return true;
//}




using TestResult = std::tuple<bool, int>;

#define ADD_TEST(x)                                                               \
if (run_all || test_names.contains(#x)) {                                         \
  auto start = std::chrono::high_resolution_clock::now();                         \
  bool passed = x();                                                              \
  auto stop = std::chrono::high_resolution_clock::now();                          \
  int duration = duration_cast<std::chrono::microseconds>(stop - start).count();  \
  tests[#x "()"] = std::make_tuple(passed, duration);                             \
}                                                                                 \

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }

  ADD_TEST(test_add_term);
  ADD_TEST(test_term_to_matrix);
  ADD_TEST(test_hamiltonian_support);
  ADD_TEST(test_to_gate);
  ADD_TEST(test_free_fermion_state);
  ADD_TEST(test_amplitudes);
  ADD_TEST(test_majorana_expectation);
  ADD_TEST(test_paulistring_expectation);
  ADD_TEST(test_bitstring_expectation);
  ADD_TEST(test_mzr);
  ADD_TEST(test_weak_mzr);
  ADD_TEST(test_majorana_to_pauli);
  ADD_TEST(test_sample_paulis);
  ADD_TEST(test_majorana_covariance_matrix);
  //ADD_TEST(test_det);
  //ADD_TEST(test_det_case);

  constexpr char green[] = "\033[1;32m";
  constexpr char black[] = "\033[0m";
  constexpr char red[] = "\033[1;31m";

  auto test_passed_str = [&](bool passed) {
    std::stringstream stream;
    if (passed) {
      stream << green << "PASSED" << black;
    } else {
      stream << red << "FAILED" << black;
    }
    
    return stream.str();
  };

  if (tests.size() == 0) {
    std::cout << "No tests to run.\n";
  } else {
    double total_duration = 0.0;
    for (const auto& [name, result] : tests) {
      auto [passed, duration] = result;
      std::cout << fmt::format("{:>40}: {} ({:.2f} seconds)\n", name, test_passed_str(passed), duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
