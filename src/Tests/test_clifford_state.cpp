#include "tests.hpp"

#include "QuantumCircuit.h"
#include "QuantumState.h"
#include "CliffordState.h"
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

template <typename... QuantumStates>
void randomize_state(QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);
  size_t depth = num_qubits;
  for (size_t i = 0; i < depth; i++) {
    uint32_t k = (i % 2) ? 0 : 1;
    for (uint32_t j = k; j < num_qubits - 1; j += 2) {
      Eigen::Matrix4cd random = Eigen::Matrix4cd::Random();
      std::vector<uint32_t> qubits = {j, j + 1};
      ([&] {
       if constexpr (std::is_same_v<std::decay_t<QuantumStates>, QuantumCircuit>) {
         states.add_gate(random, qubits);
       } else {
         states.evolve(random, qubits);
       }
      }(), ...);
    }
  }
}

template <typename... QuantumStates>
void randomize_state_haar(QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);
  size_t depth = 2;

  qc.append(generate_haar_circuit(num_qubits, depth, false));
  qc.apply(states...);
}

template <typename... QuantumStates>
void randomize_state_clifford(size_t depth, QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);

  for (size_t k = 0; k < depth; k++) {
    for (size_t i = 0; i < num_qubits/2 - 1; i++) {
      uint32_t q1 = (k % 2) ? 2*i : 2*i + 1;
      uint32_t q2 = q1 + 1;

      QuantumCircuit rc = random_clifford(2);
      rc.apply({q1, q2}, states...);
    }
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

bool test_chp_state() {
  constexpr size_t nqb = 5;
  QuantumCHPState chp(nqb);
  Statevector psi(nqb);

  for (size_t i = 0; i < 30; i++) {
    QuantumCircuit qc(nqb);
    qc.append(random_clifford(nqb));
    qc.apply(psi, chp);

    Qubits qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), 0);

    PauliString p = PauliString::randh(nqb);
    p = PauliString(nqb);
    p.set_z(0, 1);

    // Make sure measurement outcomes are the same
    Measurement m1(qubits, p);
    auto [b1, p1] = psi.measure(m1);

    Measurement m2(qubits, p, b1);
    auto [b2, p2] = chp.measure(m2);

    Statevector psi_chp = chp.to_statevector();

    ASSERT((b1 == b2) && is_close(p1, p2), "Different measurement outcomes observed.");

    ASSERT(states_close(psi, psi_chp), fmt::format("Clifford simulators disagree."));

    for (size_t k = 0; k < 100; k++) {
      PauliString P_exp = PauliString::randh(nqb);

      auto c1 = std::abs(psi.expectation(P_exp));
      auto c2 = std::abs(chp.expectation(P_exp));
      ASSERT(is_close(c1, c2), fmt::format("Expectation of {} = {:.5f}, {:.5f} is not equal.", P_exp, c1, c2));
    }
  }

  return true;
}

#ifdef __AVX__

bool test_chp_simd() {
  for (size_t i = 0; i < 10; i++) {
    size_t nqb = randi(1, 129);
    QuantumCHPState chp1(nqb, false);
    QuantumCHPState chp2(nqb, true);
    chp1.set_print_mode("paulis_all");
    chp2.set_print_mode("paulis_all");

    QuantumCircuit qc(nqb);
    qc.append(random_clifford(nqb));
    for (size_t q = 0; q < nqb/2; q++) {
      auto r = randi(0, nqb);
      qc.mzr(r);
    }

    EvolveOpts opts;
    opts.return_measurement_outcomes = true;

    int s = randi();
    Random::seed_rng(s);
    std::vector<bool> results1 = std::get<std::vector<bool>>(chp1.evolve(qc, opts).value());
    Random::seed_rng(s);
    std::vector<bool> results2 = std::get<std::vector<bool>>(chp2.evolve(qc, opts).value());

    Qubits qubits = random_qubits(nqb, randi(1, nqb - 1));
    auto e1 = chp1.entanglement(qubits, 2);
    auto e2 = chp2.entanglement(qubits, 2);
    ASSERT(is_close(e1, e2), fmt::format("Error in CHP entanglement: {} and {}\n", e1, e2));
    ASSERT(chp1 == chp2, fmt::format("States = \n{}\n\n{}\n\ns = {}\n", chp1.to_string(), chp2.to_string(), s));
  }

  return true;
}

#else

bool test_chp_simd() {
  return true;
}

#endif

bool test_async_chp() {
  int s = randi();
  std::cout << fmt::format("seed = {}\n", s);

  Random::seed_rng(s);
  for (size_t i = 0; i < 100; i++) {
    uint32_t nqb = randi(256, 1024);
    QuantumCHPState chp1(nqb);
    QuantumCHPState chp2(nqb);
    chp1.set_print_mode("paulis_all");
    chp2.set_print_mode("paulis_all");

    QuantumCircuit qc(nqb);
    for (uint32_t i = 0; i < 100; i++) {
      uint32_t r = randi() % nqb;
      qc.random_clifford({r, (r+1)%nqb});
    }

    for (size_t q = 0; q < nqb/2; q++) {
      auto r = randi(0, nqb);
      //qc.mzr(r);
    }

    EvolveOpts opts1;
    opts1.return_measurement_outcomes = true;

    EvolveOpts opts2;
    opts2.return_measurement_outcomes = true;
    opts2.async_threads = 4;

    s = randi();
    Random::seed_rng(s);
    std::vector<bool> results1 = std::get<std::vector<bool>>(chp1.evolve(qc, opts1).value());
    Random::seed_rng(s);
    std::vector<bool> results2 = std::get<std::vector<bool>>(chp2.evolve(qc, opts2).value());

    Qubits qubits = random_qubits(nqb, randi(1, nqb - 1));
    auto e1 = chp1.entanglement(qubits, 2);
    auto e2 = chp2.entanglement(qubits, 2);
    ASSERT(is_close(e1, e2), fmt::format("Error in CHP entanglement: {} and {}\n", e1, e2));
    chp1.rref();
    chp2.rref();
    ASSERT(chp1 == chp2, fmt::format("States do not match. s = {}\n", s));
    //ASSERT(chp1 == chp2, fmt::format("States = \n{}\n\n{}\n\ns = {}\n", chp1.to_string(), chp2.to_string(), s));
  }

  return true;
}

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

  ADD_TEST(test_chp_state);
  ADD_TEST(test_chp_simd);
  //ADD_TEST(test_forced_measurement);
  //ADD_TEST(test_bitstring_expectation);
  //ADD_TEST(test_measurement_record);
  //ADD_TEST(test_chp_probs);
  ADD_TEST(test_async_chp);

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
