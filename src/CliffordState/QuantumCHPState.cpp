#include "QuantumCHPState.h"
#include "ThreadPool.hpp"

#include <queue>
#include <condition_variable>
#include <mutex>

QuantumCHPState::QuantumCHPState(uint32_t num_qubits, bool use_simd) : CliffordState(num_qubits) {
  if (use_simd) {
    tableau = std::make_unique<TableauSIMD>(num_qubits);
  } else {
    tableau = std::make_unique<Tableau>(num_qubits);
  }
}

std::string QuantumCHPState::to_string() const {
  if (print_mode == 0) {
    return tableau->to_string(true);
  } else if (print_mode == 1) {
    return tableau->to_string(false);
  } else if (print_mode == 2) {
    return tableau->to_string_ops(true);
  } else {
    return tableau->to_string_ops(false);
  }
}

void QuantumCHPState::rref() {
  tableau->rref();
}

void QuantumCHPState::xrref() {
  tableau->xrref();
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
  return tableau->to_statevector();
}

//EvolveResult QuantumCHPState::evolve(const QuantumCircuit& circuit, const Qubits& qubits, EvolveOpts opts) {
//  QuantumCircuit circuit_mapped(circuit);
//  circuit_mapped.resize_qubits(num_qubits);
//  circuit_mapped.apply_qubit_map(qubits);
//  
//  return QuantumCHPState::evolve(circuit_mapped, opts);
//}
//
//EvolveResult QuantumCHPState::evolve(const QuantumCircuit& circuit, EvolveOpts opts) {
//  if (circuit.get_num_parameters() > 0) {
//    throw std::invalid_argument("Unbound QuantumCircuit parameters; cannot evolve.");
//  }
//
//  if (!circuit.is_clifford()) {
//    throw std::runtime_error("Provided circuit is not Clifford.");
//  }
//
//  if (opts.async_threads <= 1) {
//    return QuantumState::_evolve(circuit, opts);
//  }
//
//  BitString bits(circuit.get_num_cbits());
//
//  const size_t N = circuit.length();
//
//  CircuitDAG dag = circuit.to_binned_dag(binary_word_size()/2);
//  auto reversed_dag = make_reversed_dag(dag);
//
//  size_t num_measurements = circuit.get_num_measurements();
//  std::vector<MeasurementData> measurements(num_measurements);
//  std::vector<size_t> measurement_map = circuit.get_measurement_map();
//  std::map<size_t, size_t> reversed_map = reverse_map(measurement_map);
//
//  auto execute_inst = quantumcircuit_utils::overloaded {
//    [&](const QuantumInstruction& qinst, size_t i) {
//      auto result = CliffordState::evolve(qinst);
//      if (result) {
//        measurements[reversed_map.at(i)] = result.value();
//      }
//    },
//    [&](const ClassicalInstruction& clinst, size_t i) {
//      clinst.apply(bits);
//    },
//    [&](const ConditionedInstruction& cinst, size_t i) {
//      const auto& inst = circuit.instructions[i];
//      if (!cinst.should_execute(bits)) {
//        return;
//      }
//
//      auto result = CliffordState::evolve(cinst.inst);
//
//      if (result) {
//        if (cinst.target) {
//          bits.set(cinst.target.value(), result->first);
//        }
//
//        measurements[reversed_map.at(i)] = result.value();
//      }
//    }
//  };
//
//  // === 1. Compute in-degrees ===
//  std::vector<std::atomic<int>> indegree(N);
//  for (size_t i = 0; i < N; ++i) {
//    indegree[i].store(static_cast<int>(reversed_dag.degree(i)), std::memory_order_relaxed);
//  }
//
//  // === 2. Find initial ready tasks ===
//  std::queue<size_t> ready;
//  for (size_t i = 0; i < N; ++i) {
//    if (indegree[i] == 0) {
//      ready.push(i);
//    }
//  }
//
//  // === 3. Shared synchronization ===
//  std::mutex mtx;
//  std::condition_variable cv;
//  std::atomic<size_t> completed = 0;
//
//  // === 4. Function to submit a task ===
//  ThreadPool pool(opts.async_threads);
//  auto submit_task = [&](size_t id) {
//    pool.submit([&, id] {
//      // run the task and store the result
//      std::visit([&](
//        auto& inst) {
//          execute_inst(inst, id);
//        }, dag.get_val(id)
//      );
//
//      // mark children
//      for (size_t child : dag.neighbors(id)) {
//        int old = indegree[child].fetch_sub(1) - 1;
//        if (old == 0) {
//          std::unique_lock lk(mtx);
//          ready.push(child);
//          cv.notify_one();
//        }
//      }
//
//      // if all tasks finished, wake the main thread
//      if (++completed == N) {
//        std::unique_lock lk(mtx);
//        cv.notify_all();
//      }
//    });
//  };
//
//  // === 5. Main scheduler loop ===
//  {
//    std::unique_lock lk(mtx);
//    while (completed.load() < N) {
//      // submit all currently ready tasks
//      while (!ready.empty()) {
//        size_t id = ready.front();
//        ready.pop();
//        submit_task(id);
//      }
//
//      // sleep until:
//      //  - a new task becomes ready OR
//      //  - all tasks finish
//      cv.wait(lk, [&] {
//        return completed.load() == N || !ready.empty();
//      });
//    }
//  }
//
//  return process_measurement_results(measurements, opts);
//}

void QuantumCHPState::h(uint32_t a) {
  tableau->h(a);
}

void QuantumCHPState::s(uint32_t a) {
  tableau->s(a);
}

void QuantumCHPState::sd(uint32_t a) {
  tableau->s(a);
  tableau->s(a);
  tableau->s(a);
}

void QuantumCHPState::cx(uint32_t a, uint32_t b) {
  tableau->cx(a, b);
}

void QuantumCHPState::cy(uint32_t a, uint32_t b) {
  tableau->s(b);
  tableau->h(b);
  tableau->cz(a, b);
  tableau->h(b);
  tableau->sd(b);
}

void QuantumCHPState::cz(uint32_t a, uint32_t b) {
  tableau->h(b);
  tableau->cx(a, b);
  tableau->h(b);
}

PauliString QuantumCHPState::get_stabilizer(size_t i) const {
  return tableau->get_stabilizer(i);
}

PauliString QuantumCHPState::get_destabilizer(size_t i) const {
  std::vector<Pauli> paulis(num_qubits);
  for (size_t j = 0; j < num_qubits; j++) {
    paulis[i] = tableau->get_pauli(i, j);
  }
  uint8_t phase = tableau->get_phase(i);
  return PauliString(paulis, phase);
}

double QuantumCHPState::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  if (support) {
    // TODO add support for SIMD
    //Tableau restricted = tableau->partial_trace(to_qubits(support_complement(support.value(), num_qubits)));
    return tableau->bitstring_amplitude(bits);
  } else {
    return tableau->bitstring_amplitude(bits);
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
  std::vector<PauliString> stabilizers(num_qubits);
  for (size_t i = 0; i < num_qubits; i++) {
    stabilizers[i] = get_stabilizer(i);
  }
  return stabilizers;
}

void QuantumCHPState::random_clifford(const Qubits& qubits) {
  random_clifford_impl(qubits, *this);
}

double QuantumCHPState::mzr_expectation(uint32_t a) const {
  auto [deterministic, _] = tableau->mzr_deterministic(a);
  if (!deterministic) {
    return 0.0;
  } else {
    PauliString scratch(num_qubits);
    for (uint32_t i = 0; i < num_qubits; i++) {
      Pauli p = tableau->get_pauli(i, a);
      if (p == Pauli::X || p == Pauli::Y) {
        scratch = scratch * tableau->get_stabilizer(i);
      }
    }

    // TODO check
    bool b = scratch.get_r();
    return 2*int(b) - 1.0;
  }
}

MeasurementData QuantumCHPState::mzr(uint32_t a, std::optional<bool> outcome) {
  return tableau->mzr(a, outcome);
}

double QuantumCHPState::sparsity() const {
  return tableau->sparsity();
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

  int rank = tableau->rank(qubits);

  int s = rank - partition_size;

  return static_cast<double>(s);
}

int QuantumCHPState::xrank() const {
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return tableau->xrank(qubits);
}

int QuantumCHPState::partial_xrank(const Qubits& qubits) const {
  return tableau->xrank(qubits);
}

int QuantumCHPState::rank() const {
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return tableau->rank(qubits);
}

int QuantumCHPState::partial_rank(const Qubits& qubits) const {
  return tableau->rank(qubits);
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
    "num_qubits", &Tableau::num_qubits
    // TODO add serialization back in
    //"stabilizers", &Tableau::stabilizers,
    //"destabilizers", &Tableau::destabilizers
  );
};

template<>
struct glz::meta<QuantumCHPState> {
  static constexpr auto value = glz::object(
    //"tableau", &QuantumCHPState::tableau,
    "print_mode", &QuantumCHPState::print_mode
  );
};

DEFINE_SERIALIZATION(QuantumCHPState);
