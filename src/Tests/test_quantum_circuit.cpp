#include "tests.hpp"

#include "QuantumCircuit.h"
#include "QuantumState.h"

Qubits random_qubits(size_t num_qubits, size_t k) {
  std::minstd_rand rng(randi());
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  std::shuffle(qubits.begin(), qubits.end(), rng);

  Qubits r(qubits.begin(), qubits.begin() + k);
  return r;
}

bool test_circuit_dag() {
  constexpr size_t nqb = 8;
  QuantumCircuit qc(nqb);
  qc.add_gate(haar_unitary(2), {0, 1});
  qc.add_gate(haar_unitary(2), {2, 3});
  qc.add_gate(haar_unitary(2), {1, 2});
  qc.mzr(0);

  CircuitDAG dag = qc.to_dag();

  std::vector<std::set<size_t>> expected = {
    {2, 3},
    {2},
    {},
    {},
  };

  for (size_t i = 0; i < qc.length(); i++) {
    for (size_t j = 0; j < qc.length(); j++) {
      if (expected[i].contains(j)) {
        ASSERT(dag.contains_edge(i, j));
      } else {
        ASSERT(!dag.contains_edge(i, j));
      }
    }
  }

  return true;
}

QuantumCircuit random_unitary_circuit(size_t nqb, size_t depth, const std::vector<size_t>& gate_sizes) {
  QuantumCircuit qc(nqb);
  for (size_t i = 0; i < depth; i++) {
    size_t r = gate_sizes[randi(0, gate_sizes.size())];
    size_t q = randi(0, nqb - r + 1);
    Qubits qubits(r);
    std::iota(qubits.begin(), qubits.end(), q);
    qc.add_gate(haar_unitary(r), qubits);
  }

  return qc;
}

bool test_qc_canonical() {
  constexpr size_t nqb = 6;

  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc = random_unitary_circuit(nqb, 10, {2});
    CircuitDAG dag = qc.to_dag();
    QuantumCircuit canon = QuantumCircuit::to_circuit(dag, nqb, 0, randf() < 0.5);
    ASSERT(qc.to_matrix().isApprox(canon.to_matrix()));
  }

  return true;
}

bool test_pauli_reduce() {
  for (size_t i = 0; i < 100; i++) {
    size_t nqb =  randi() % 20 + 1;
    PauliString p1 = PauliString::randh(nqb);
    PauliString p2 = PauliString::randh(nqb);
    while (p2.commutes(p1)) {
      p2 = PauliString::randh(nqb);
    }

    std::vector<uint32_t> qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), 0);
    QuantumCircuit qc(nqb);
    reduce_paulis(p1, p2, qubits, qc);

    PauliString p1_ = p1;
    PauliString p2_ = p2;
    qc.apply(p1_, p2_);

    ASSERT(p1_ == PauliString::basis(nqb, "X", 0, false) && p2_ == PauliString::basis(nqb, "Z", 0, false),
        fmt::format("p1 = {} and p2 = {}\nreduced to {} and {}.", p1, p2, p1_, p2_));
  }

  return true;
}

bool test_dag_to_circuit() {
  constexpr size_t nqb = 6;
  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc = random_unitary_circuit(nqb, 10, {1, 2, 3});

    CircuitDAG dag = qc.to_dag();
    QuantumCircuit left = QuantumCircuit::to_circuit_left_to_right(dag, nqb, 0);
    QuantumCircuit right = QuantumCircuit::to_circuit_right_to_left(dag, nqb, 0);
    ASSERT(qc.to_matrix().isApprox(left.to_matrix()));
    ASSERT(qc.to_matrix().isApprox(right.to_matrix()));
  }
}

bool test_qc_simplify() {
  constexpr size_t nqb = 6;

  for (size_t i = 0; i < 20; i++) {
    QuantumCircuit qc = random_unitary_circuit(nqb, 20, {1, 2});
    QuantumCircuit simple = qc.simplify(randf() < 0.5);

    ASSERT(qc.to_matrix().isApprox(simple.to_matrix()));
  }

  return true;
}

bool test_qc_reduce() {
  constexpr size_t nqb = 8;
  QuantumCircuit qc(nqb);
  auto qubits = random_qubits(nqb, 4);
  PauliString p = PauliString::randh(4);
  qc.h(0);
  qc.h(5);
  qc.add_measurement(qubits, p);
  qc.cx(0, 2);
  qc.swap(6, 2);
  qc.x(1);
  Qubits support = qc.get_support();
  std::set<uint32_t> exps(qubits.begin(), qubits.end());
  exps.insert(0);
  exps.insert(1);
  exps.insert(2);
  exps.insert(5);
  exps.insert(6);
  Qubits expected(exps.begin(), exps.end());
  std::sort(expected.begin(), expected.end());
  ASSERT(support == expected);

  Qubits map = reduced_support(support, nqb);
  QuantumCircuit qc_(qc);
  qc_.apply_qubit_map(map);
  qc_.resize_qubits(support.size());
  QuantumCircuit qc_r(nqb);
  qc_r.append(qc_, support);

  std::string s1 = qc.to_string();
  std::string s2 = qc_r.to_string();
  ASSERT(s1 == s2);
  // TODO check to_matrix equality?

  auto components = qc.split_into_unitary_components();
  for (const auto &q : components) {
    bool is_unitary = instruction_is_unitary(q.instructions[0]);
    for (size_t k = 1; k < q.length(); k++) {
      ASSERT(instruction_is_unitary(q.instructions[k]) == is_unitary);
    }
  }

  return true;
}

bool test_pauli() {
  Pauli id = Pauli::I;
  Pauli x = Pauli::X;
  Pauli y = Pauli::Y;
  Pauli z = Pauli::Z;

  auto validate_result = [](Pauli p1, Pauli p2, char g, uint8_t p) {
    auto [result, phase] = multiply_pauli(p1, p2);
    return (g == pauli_to_char(result)) && (p == phase);
  };

  ASSERT(pauli_to_char(id) == 'I');
  ASSERT(pauli_to_char(x) == 'X');
  ASSERT(pauli_to_char(y) == 'Y');
  ASSERT(pauli_to_char(z) == 'Z');

  ASSERT(validate_result(id, id, 'I', 0));
  ASSERT(validate_result(id, x, 'X', 0));
  ASSERT(validate_result(x, id, 'X', 0));
  ASSERT(validate_result(id, y, 'Y', 0));
  ASSERT(validate_result(y, id, 'Y', 0));
  ASSERT(validate_result(id, z, 'Z', 0));
  ASSERT(validate_result(z, id, 'Z', 0));
  ASSERT(validate_result(x, x, 'I', 0));
  ASSERT(validate_result(x, y, 'Z', 1));
  ASSERT(validate_result(y, x, 'Z', 3));
  ASSERT(validate_result(x, z, 'Y', 3));
  ASSERT(validate_result(z, x, 'Y', 1));
  ASSERT(validate_result(y, z, 'X', 1));
  ASSERT(validate_result(z, y, 'X', 3));

  return true;
}

bool test_parametrized_circuit() {
  constexpr size_t nqb = 6;


  for (size_t i = 0; i < 10; i++) {
    size_t num_gates = 10;
    std::vector<double> parameters(num_gates);

    QuantumCircuit qc1(nqb);
    QuantumCircuit qc2(nqb);
    for (size_t j = 0; j < num_gates; j++) {
      parameters[j] = randf() * M_PI;
      double p = randf();
      if (p < 0.25) {
        uint32_t q = randi(0, nqb);
        qc1.rx(q);
        qc2.rx(q, parameters[j]);
      } else if (p < 0.5) {
        uint32_t q = randi(0, nqb);
        qc1.ry(q);
        qc2.ry(q, parameters[j]);
      } else if (p < 0.75) {
        uint32_t q = randi(0, nqb);
        qc1.rz(q);
        qc2.rz(q, parameters[j]);
      } else {
        size_t n = randi(1, 4);
        Qubits qubits = random_qubits(nqb, n);
        PauliString pauli = PauliString::randh(n);
        qc1.rp(qubits, pauli);
        qc2.rp(qubits, pauli, parameters[j]);
      }
    }

    QuantumCircuit qc1_ = qc1.bind_params(parameters);

    ASSERT(qc1_.to_matrix().isApprox(qc2.to_matrix()));
    ASSERT(qc1_.to_matrix().adjoint().isApprox(qc2.adjoint().to_matrix()));
  }

  return true;
}

bool test_conditioned_operation() {
  constexpr size_t nqb = 10;

  for (size_t i = 0; i < 20; i++) {
    QuantumCircuit qc(nqb, nqb);
    for (size_t q = 0; q < nqb; q++) {
      qc.h(q);
      qc.mzr(q, q);
      qc.add_gate("x", {q}, q);
    }

    EvolveOpts opts;
    opts.return_measurement_outcomes = true;
    opts.simplify_circuit = true;

    Statevector psi(nqb);
    auto results = std::get<std::vector<bool>>(psi.evolve(qc, opts).value());

    Statevector psi0(nqb);

    ASSERT(is_close(std::abs(psi.inner(psi0)), 1.0));
  }

  return true;
}

bool test_random_conditioned_operation() {
  constexpr size_t nqb = 10;
  
  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc(nqb, nqb);

    for (size_t d = 0; d < 10; d++) {
      if (randf() < 0.8) { // Unitary
        size_t n = randi(1, 4);
        PauliString P = PauliString::randh(n);
        double theta = randf(0, 2*M_PI);

        Qubits qubits(n);
        std::iota(qubits.begin(), qubits.end(), randi(0, nqb - n - 1));
        size_t control = randi(0, nqb);
        qc.rp(qubits, P, theta, control);
      } else {
        size_t n = randi(1, 4);
        PauliString P = PauliString::randh(n);
        double theta = randf(0, 2*M_PI);

        Qubits qubits(n);
        std::iota(qubits.begin(), qubits.end(), randi(0, nqb - n - 1));
        size_t target = randi(0, nqb);
        qc.add_measurement(qubits, P, std::nullopt, target);
      }
    }

    EvolveOpts opts;
    opts.return_measurement_outcomes = true;
    opts.simplify_circuit = true;

    // TODO ensure that outcomes are matched to correct measurements, even when order is not preserved 
    // due to DAG contraction/reordering 
    Statevector psi1(nqb);
    auto results1 = std::get<std::vector<bool>>(psi1.evolve(qc, opts).value());

    qc.set_measurement_outcomes(results1);

    Statevector psi2(nqb);
    auto results2 = std::get<std::vector<bool>>(psi2.evolve(qc, opts).value());
  }


  return true;
}

bool test_simplify_cbits() {
  constexpr size_t nqb = 4;
  QuantumCircuit qc(nqb, nqb);
  for (size_t k = 0; k < 10; k++) {
    qc.add_gate("x", {randi(0, nqb)}, {0});
  }

  QuantumCircuit simple = qc.simplify(randf() < 0.5);

  // All gates depend on each other causally through 0th classical bit. Do not allow
  // for any rearrangement
  ASSERT(qc.to_string() == simple.to_string());

  return true;
}

double objective(const std::vector<double>& params) {
  return std::pow(params[0] - 3.0, 2);
}

// Gradient of f(x): df/dx = 2(x - 3)
std::vector<double> gradient(const std::vector<double>& params) {
  return { 2.0 * (params[0] - 3.0) };
}

bool test_adam() {
  // Initialize optimizer
  ADAMOptimizer opt;

  // Start with parameter far from optimum
  std::vector<double> params = {0.0};

  std::cout << "Initial value: f(" << params[0] << ") = " 
    << objective(params) << "\n";

  // Run optimization for 100 steps
  for (int step = 1; step <= 1000; step++) {
    auto grads = gradient(params);
    params = opt.step(params, grads);

    if (step % 10 == 0) {
      std::cout << "Step " << step 
        << " | x = " << params[0] 
        << " | f(x) = " << objective(params) 
        << "\n";
    }
  }

  std::cout << "Final result: x ≈ " << params[0] 
    << ", f(x) ≈ " << objective(params) << "\n";

  return true;
}

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }

  ADD_TEST(test_circuit_dag);
  ADD_TEST(test_qc_reduce);
  ADD_TEST(test_pauli_reduce);
  ADD_TEST(test_qc_canonical);
  ADD_TEST(test_qc_simplify);
  ADD_TEST(test_pauli);
  ADD_TEST(test_parametrized_circuit);
  ADD_TEST(test_conditioned_operation);
  //ADD_TEST(test_random_conditioned_operation);
  ADD_TEST(test_adam);
  ADD_TEST(test_simplify_cbits);

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
