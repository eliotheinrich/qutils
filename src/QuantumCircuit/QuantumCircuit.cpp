#include "QuantumCircuit.h"
#include "Clifford.hpp"
#include "Instructions.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <set>
#include <queue>

std::string QuantumCircuit::to_string() const {
	std::string s = "";
	for (auto const &inst : instructions) {
    s += fmt::format("{}\n", inst);
	}

	return s;
}

uint32_t QuantumCircuit::num_params() const {
	uint32_t n = 0;
	for (auto const &inst : instructions) {
    n += std::visit(quantumcircuit_utils::overloaded {
      [](std::shared_ptr<Gate> gate) -> uint32_t { 
        return gate->num_params(); 
      },
      [](const FreeFermionGate& gate) -> uint32_t { 
        return gate.num_params(); 
      },
      [](const Measurement& m) -> uint32_t { 
        return 0u; 
      },
      [](const WeakMeasurement& m) -> uint32_t { 
        return m.num_params(); 
      }
    }, inst.inst);
	}
	
	return n;
}

bool QuantumCircuit::is_clifford() const {
  for (auto const& inst : instructions) {
    bool valid = std::visit(quantumcircuit_utils::overloaded {
			[](std::shared_ptr<Gate> gate) -> bool { 
        return gate->is_clifford(); 
      },
      [](const FreeFermionGate& gate) -> bool { 
        return gate.is_clifford(); 
      },
			[](const Measurement &m) -> bool { 
        return true; 
      },
			[](const WeakMeasurement& m) -> bool { 
        return false; 
      }
		}, inst.inst);
    
    if (!valid) {
      return false;
    }
  }

  return true;
}

uint32_t QuantumCircuit::length() const {
  return instructions.size();
}

bool QuantumCircuit::is_unitary() const {
  for (auto const &inst : instructions) {
    if (!instruction_is_unitary(inst.inst)) {
      return false;
    }
  }

  return true;
}

CircuitDAG QuantumCircuit::to_dag() const {
  CircuitDAG dag(length());

  std::vector<std::queue<size_t>> covers(num_qubits + num_cbits);
  std::vector<Qubits> supports(length());
  std::vector<Qubits> classical_supports(length());

  for (size_t i = 0; i < length(); i++) {
    const ConditionedInstruction& cinst = instructions[i];
    dag.set_val(i, copy_instruction(cinst));

    supports[i] = get_instruction_support(cinst);

    for (uint32_t q : supports[i]) {
      covers[q].push(i);
    }

    classical_supports[i] = get_instruction_classical_support(cinst);

    for (uint32_t q : classical_supports[i]) {
      covers[q + num_qubits].push(i);
    }
  }

  for (size_t i = 0; i < length(); i++) {
    // go through every qubit in the support of instruction i
    // if any other instructions act on this qubit, then that instruction
    // causally depends on instruction i
    for (uint32_t q : supports[i]) {
      covers[q].pop();
      if (covers[q].size() > 0) {
        // next node that depends on i
        size_t j = covers[q].front(); 

        if (j != i) {
          dag.add_edge(i, j);
        }
      }
    }

    for (uint32_t q_ : classical_supports[i]) {
      uint32_t q = q_ + num_qubits;
      covers[q].pop();
      if (covers[q].size() > 0) {
        // next node that depends on i
        size_t j = covers[q].front(); 

        if (j != i) {
          dag.add_edge(i, j);
        }
      }
    }
  }

  return dag;
}

using TreeEntry = std::pair<size_t, uint32_t>;

// Comparator: sort pairs <nodeIndex, leftmostQubit>
struct PairCmp {
  bool operator()(auto const& a, auto const& b) const {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  }
};

DirectedGraph<int> make_reversed_dag(const CircuitDAG& dag) {
  DirectedGraph<int> reversed_dag(dag.num_vertices);
  for (size_t i = 0; i < dag.num_vertices; i++) {
    for (const auto& j : dag.edges_of(i)) {
      reversed_dag.add_edge(j, i);
    }
  }

  return reversed_dag;
}

QuantumCircuit QuantumCircuit::to_circuit(const CircuitDAG& dag, uint32_t num_qubits, uint32_t num_cbits, bool ltr) {
  if (ltr) {
    return to_circuit_left_to_right(dag, num_qubits, num_cbits);
  } else {
    return to_circuit_right_to_left(dag, num_qubits, num_cbits);
  }
}

QuantumCircuit QuantumCircuit::to_circuit_left_to_right(const CircuitDAG& dag, uint32_t num_qubits, uint32_t num_cbits) {
  auto reversed_dag = make_reversed_dag(dag);

  // Hold a pair of the DAG index and the leftmost qubit
  std::set<TreeEntry, PairCmp> leafs;
  for (size_t i = 0; i < dag.num_vertices; i++) {
    if (reversed_dag.degree(i) == 0) {
      const ConditionedInstruction& cinst = dag.get_val(i);
      uint32_t q = std::ranges::min(get_instruction_support(cinst));
      leafs.emplace(i, q);
    }
  }

  QuantumCircuit circuit(num_qubits, num_cbits);
  size_t i = 0;
  int pos = 0;
  std::set<size_t> visited;
  bool first = true;
  while (!leafs.empty()) {
    auto it = leafs.begin();
    if (!first) {
      TreeEntry key = {SIZE_MAX, pos};
      auto it = leafs.lower_bound(key);

      uint32_t best = UINT32_MAX;

      auto try_pick = [&](auto iter) {
        int q = iter->second;
        uint32_t d = (q > pos ? q - pos : pos - q);
        if (d < best) {
          best = d;
          it = iter;
        }
      };

      if (it != leafs.end()) {
        try_pick(it);
      } 
      if (it != leafs.begin()) {
        try_pick(std::prev(it));
      }
    }

    first = false;

    std::tie(i, pos) = *it;

    visited.insert(i);
    circuit.add_instruction(copy_instruction(dag.get_val(i)));

    std::set<size_t> new_leafs;
    for (size_t j : dag.edges_of(i)) {
      bool include = true;
      for (size_t k : reversed_dag.edges_of(j)) {
        if (!visited.contains(k)) {
          include = false;
          break;
        }
      }

      if (include) {
        new_leafs.insert(j);
      }
    }

    leafs.erase(it);

    for (size_t j : new_leafs) {
      const ConditionedInstruction& cinst = dag.get_val(j);
      uint32_t q = std::ranges::min(get_instruction_support(cinst));
      leafs.emplace(j, q);
    }
  }

  return circuit;
}

QuantumCircuit QuantumCircuit::to_circuit_right_to_left(const CircuitDAG& dag, uint32_t num_qubits, uint32_t num_cbits) {
  auto reversed_dag = make_reversed_dag(dag);

  // Hold a pair of the DAG index and the leftmost qubit
  std::set<TreeEntry, PairCmp> leafs;
  for (size_t i = 0; i < dag.num_vertices; i++) {
    if (reversed_dag.degree(i) == 0) {
      const ConditionedInstruction& cinst = dag.get_val(i);
      uint32_t q = std::ranges::max(get_instruction_support(cinst));
      leafs.emplace(i, q);
    }
  }

  QuantumCircuit circuit(num_qubits, num_cbits);
  size_t i = 0;
  int pos = num_qubits;
  std::set<size_t> visited;
  bool first = true;
  while (!leafs.empty()) {
    auto it = leafs.begin();
    if (!first) {
      TreeEntry key = {SIZE_MAX, pos};
      auto it = leafs.upper_bound(key);

      uint32_t best = UINT32_MAX;

      auto try_pick = [&](auto iter) {
        int q = iter->second;
        uint32_t d = (q > pos ? q - pos : pos - q);
        if (d < best) {
          best = d;
          it = iter;
        }
      };

      if (it != leafs.end()) {
        try_pick(it);
      } 
      if (it != leafs.begin()) {
        try_pick(std::prev(it));
      }
    }

    first = false;

    std::tie(i, pos) = *it;

    visited.insert(i);
    circuit.add_instruction(copy_instruction(dag.get_val(i)));

    std::set<size_t> new_leafs;
    for (size_t j : dag.edges_of(i)) {
      bool include = true;
      for (size_t k : reversed_dag.edges_of(j)) {
        if (!visited.contains(k)) {
          include = false;
          break;
        }
      }

      if (include) {
        new_leafs.insert(j);
      }
    }

    leafs.erase(it);

    for (size_t j : new_leafs) {
      const ConditionedInstruction& cinst = dag.get_val(j);
      uint32_t q = std::ranges::max(get_instruction_support(cinst));
      leafs.emplace(j, q);
    }
  }

  return circuit;
}

std::optional<std::pair<size_t, size_t>> find_mergeable(const CircuitDAG& dag, const auto& reversed_dag) {
  auto mergeable = [](const ConditionedInstruction& cinst) {
    return instruction_is_unitary(cinst) && !cinst.target && !cinst.control;
  };

  for (size_t i = 0; i < dag.num_vertices; i++) {
    const ConditionedInstruction& cinst1 = dag.get_val(i);
    if (!mergeable(cinst1)) {
      continue;
    }
    for (size_t j : dag.edges_of(i)) {
      const ConditionedInstruction& cinst2 = dag.get_val(j);
      if (!mergeable(cinst2)) {
        continue;
      }

      Qubits s1 = get_instruction_support(cinst1);
      Qubits s2 = get_instruction_support(cinst2);

      if (reversed_dag.degree(j) == 1 && std::includes(s1.begin(), s1.end(), s2.begin(), s2.end())) {
        return std::make_pair(i, j);
      } else if (dag.degree(i) == 1 && std::includes(s2.begin(), s2.end(), s1.begin(), s1.end())) {
        return std::make_pair(i, j);
      }
    }
  }

  return std::nullopt;
}

QuantumCircuit QuantumCircuit::simplify(bool ltr) const {
  CircuitDAG dag = to_dag();
  auto reversed_dag = make_reversed_dag(dag);

  bool merged_any = true;

  while (merged_any) {
    merged_any = false;

    auto merged_pair = find_mergeable(dag, reversed_dag);
    if (!merged_pair) {
      break;
    }

    merged_any = true;
    auto [i, j] = merged_pair.value();
    const ConditionedInstruction& cinst1 = dag.get_val(i);
    const ConditionedInstruction& cinst2 = dag.get_val(j);

    QuantumCircuit qc(num_qubits, num_cbits);
    qc.add_instruction(cinst1);
    qc.add_instruction(cinst2);

    Qubits support = qc.get_support();
    Qubits map = reduced_support(support, num_qubits);
    qc.apply_qubit_map(map);
    qc.resize_qubits(support.size());

    ConditionedInstruction combined = {std::make_shared<MatrixGate>(qc.to_matrix(), support), std::nullopt, std::nullopt};

    dag.set_val(i, combined);

    std::vector<size_t> edges(dag.edges_of(j).begin(), dag.edges_of(j).end());
    std::vector<size_t> reversed_edges(reversed_dag.edges_of(j).begin(), reversed_dag.edges_of(j).end());

    for (size_t k : edges) {
      dag.add_edge(i, k);
      reversed_dag.add_edge(k, i);
    }
    for (size_t k : reversed_edges) {
      if (k == i) {
        continue;
      }

      dag.add_edge(k, i);
      reversed_dag.add_edge(i, k);
    }

    dag.remove_vertex(j);
    reversed_dag.remove_vertex(j);
  }

  QuantumCircuit circuit = to_circuit(dag, num_qubits, num_cbits, ltr);
  return circuit;
}

void QuantumCircuit::apply_qubit_map(const Qubits& qubits) {
  for (auto& cinst : instructions) {
		std::visit(quantumcircuit_utils::overloaded {
			[&qubits](std::shared_ptr<Gate> gate) {
        Qubits _qubits(gate->num_qubits);
        for (size_t q = 0; q < gate->num_qubits; q++) {
          _qubits[q] = qubits[gate->qubits[q]];
        }

        gate->qubits = _qubits;
			},
      [&qubits](FreeFermionGate& gate) {
        gate.apply_qubit_map(qubits);
      },
			[&qubits](Measurement& m) { 
        Qubits _qubits(m.qubits.size());
        for (size_t q = 0; q < m.qubits.size(); q++) {
          _qubits[q] = qubits[m.qubits[q]];
        }

        m.qubits = _qubits;
      },
      [&qubits](WeakMeasurement& m) {
        Qubits _qubits(m.qubits.size());
        for (size_t q = 0; q < m.qubits.size(); q++) {
          _qubits[q] = qubits[m.qubits[q]];
        }

        m.qubits = _qubits;
      }
		}, cinst.inst);
  }
}

void QuantumCircuit::apply_cbit_map(const Qubits& bits) {
  for (auto& cinst : instructions) {
    if (cinst.target) {
      cinst.target = bits[cinst.target.value()];
    }

    if (cinst.control) {
      cinst.control = bits[cinst.control.value()];
    }
  }
}

Qubits QuantumCircuit::get_support() const {
  std::set<uint32_t> support;

  for (auto const& inst : instructions) {
    Qubits qubits = get_instruction_support(inst);
    for (uint32_t q : qubits) {
      support.insert(q);
    }
  }

  Qubits support_(support.begin(), support.end());
  std::sort(support_.begin(), support_.end());
  return support_;
}

void QuantumCircuit::validate_instruction(const ConditionedInstruction& cinst) const {
  size_t num_qubits = this->num_qubits;
  auto validate_qubits = [num_qubits](const Qubits& qubits) {
    for (const auto q : qubits) {
      if (q >= num_qubits) {
        throw std::runtime_error(fmt::format("Invalid qubit {} passed to QuantumCircuit with {} qubits.", q, num_qubits));
      }
    }
  };

  std::visit(quantumcircuit_utils::overloaded {
    [&](std::shared_ptr<Gate> gate) {
      validate_qubits(gate->qubits);
    },
    [&](const FreeFermionGate& gate) { 
      validate_qubits(gate.get_support());
    },
    [&](const Measurement& m) { 
      validate_qubits(m.qubits);
    },
    [&](const WeakMeasurement& m) {
      validate_qubits(m.qubits);
    }
  }, cinst.inst);

  if (cinst.control && cinst.control.value() >= num_cbits) {
    throw std::runtime_error(fmt::format("Invalid control bit {} passed to QuantumCircuit with {} cbits.", cinst.control.value(), num_cbits));
  }

  if (cinst.target && cinst.target.value() >= num_cbits) {
    throw std::runtime_error(fmt::format("Invalid target bit {} passed to QuantumCircuit with {} cbits.", cinst.target.value(), num_cbits));
  }
}

void QuantumCircuit::add_instruction(const ConditionedInstruction& cinst) {
  validate_instruction(cinst);
  instructions.push_back(cinst);
}

void QuantumCircuit::add_measurement(const Measurement& m, TargetOpt target) {
  add_instruction(m, std::nullopt, target);
}

void QuantumCircuit::add_weak_measurement(const WeakMeasurement& m, TargetOpt target) {
  add_instruction(m, std::nullopt, target);
}

void QuantumCircuit::add_gate(const FreeFermionGate& gate, ControlOpt control) {
  add_instruction(gate, control, std::nullopt);
}

void QuantumCircuit::add_gate(const std::shared_ptr<Gate> &gate, ControlOpt control) {
  add_instruction(gate, control, std::nullopt);
}

void QuantumCircuit::add_gate(const std::string& name, const Qubits& qubits, ControlOpt control) {
  add_gate(std::shared_ptr<Gate>(new SymbolicGate(name, qubits)), control);
}

void QuantumCircuit::add_gate(const Eigen::MatrixXcd& gate, const Qubits& qubits, ControlOpt control) {
  if (!(gate.rows() == (1u << qubits.size()) && gate.cols() == (1u << qubits.size()))) {
    throw std::invalid_argument("Provided matrix does not have proper dimensions for number of qubits in circuit.");
  }

  add_gate(std::make_shared<MatrixGate>(gate, qubits), control);
}

void QuantumCircuit::add_gate(const Eigen::Matrix2cd& gate, uint32_t qubit, ControlOpt control) {
  Qubits qubits{qubit};
  add_gate(gate, qubits, control);
}

void QuantumCircuit::append(const QuantumCircuit& other) {
  if (num_qubits != other.num_qubits) {
    throw std::invalid_argument("Cannot append QuantumCircuits; numbers of qubits do not match.");
  }
  for (auto const &inst : other.instructions) {
    add_instruction(copy_instruction(inst));
  }
}

void QuantumCircuit::append(const QuantumCircuit& other, const Qubits& qubits) {
  if (qubits.size() != other.num_qubits) {
    throw std::invalid_argument("Cannot append QuantumCircuits; numbers of qubits do not match.");
  }

  QuantumCircuit qc_extended(other);
  qc_extended.resize_qubits(num_qubits);
  qc_extended.apply_qubit_map(qubits);

  append(qc_extended);
}

QuantumCircuit QuantumCircuit::bind_params(const std::vector<double>& params) const {
  if (params.size() != num_params()) {
    throw std::invalid_argument("Invalid number of parameters passed to bind_params.");
  }

  QuantumCircuit qc(num_qubits);

  uint32_t n = 0;
  for (auto const &inst : instructions) {
		std::visit(quantumcircuit_utils::overloaded {
      [&qc, &n, &params](std::shared_ptr<Gate> gate) {
        size_t N = gate->num_params();
        std::vector<double> gate_params(N);

        for (uint32_t i = 0; i < N; i++) {
          gate_params[i] = params[i + n];
        }
			
        n += N;
        qc.add_gate(gate->define(gate_params), gate->qubits);
      },
      [&qc, &n, &params](const FreeFermionGate& gate) {
        size_t N = gate.num_params();
        std::vector<double> gate_params(N);

        for (uint32_t i = 0; i < N; i++) {
          gate_params[i] = params[i + n];
        }

        n += N;
        qc.add_gate(gate.bind_params(gate_params));
      },
      [&qc](const Measurement& m) { 
        qc.add_measurement(m); 
      },
      [&qc, &n, &params](const WeakMeasurement& m) { 
        size_t N = m.num_params();
        std::vector<double> gate_params(N);
        for (uint32_t i = 0; i < N; i++) {
          gate_params[i] = params[i + n];
        }

        n += N;
        qc.add_weak_measurement(m.bind_params(gate_params)); 
      }
    }, inst.inst);
  }

  return qc;
}

size_t QuantumCircuit::get_num_measurements() const {
  size_t n = 0;
  for (auto const& inst : instructions) {
		std::visit(quantumcircuit_utils::overloaded {
      [](std::shared_ptr<Gate> gate) { },
      [](const FreeFermionGate& gate) { },
      [&](const Measurement& m) { n++; },
      [&](const WeakMeasurement& m) { n++; }
    }, inst.inst);
  }

  return n;
}

void QuantumCircuit::set_measurement_outcomes(const std::vector<bool>& outcomes) {
  size_t num_measurements = get_num_measurements();
  if (outcomes.size() != num_measurements) {
    throw std::runtime_error(fmt::format("Passed {} measurement outcomes to a circuit with {} measurements.", outcomes.size(), num_measurements));
  }

  size_t n = 0;
  for (auto& inst : instructions) {
		std::visit(quantumcircuit_utils::overloaded {
      [](std::shared_ptr<Gate> gate) { },
      [](const FreeFermionGate& gate) { },
      [&](Measurement& m) { m.outcome = outcomes[n++]; },
      [&](WeakMeasurement& m) { m.outcome = outcomes[n++]; }
    }, inst.inst);
  }
}

void QuantumCircuit::random_clifford(const Qubits& qubits) {
  random_clifford_impl(qubits, *this);
}

QuantumCircuit QuantumCircuit::adjoint(const std::optional<std::vector<double>>& params_opt) const {
  bool params_passed = params_opt.has_value() && params_opt.value().size() != 0;

  if (params_passed) { // Params passed; check that they are valid and then perform adjoint.
    auto params = params_opt.value();
    if (params.size() != num_params()) {
      throw std::invalid_argument("Unbound parameters; adjoint cannot be defined.");
    }

    QuantumCircuit qc = bind_params(params);
    return qc.adjoint();
  } else if (!params_passed && num_params() == 0) { // No parameters to bind; go ahead and build adjoint
    QuantumCircuit qc(num_qubits);

    for (uint32_t i = 0; i < instructions.size(); i++) {
      qc.add_instruction(instructions[instructions.size() - i - 1].adjoint());
    }

    return qc;
  } else {
    throw std::invalid_argument("Params passed but nothing to bind.");
  }
}

QuantumCircuit QuantumCircuit::reverse() const {
  QuantumCircuit qc(num_qubits);
  for (uint32_t i = 0; i < instructions.size(); i++) {
    qc.add_instruction(instructions[instructions.size() - i - 1]);
  }
  return qc;
}

QuantumCircuit QuantumCircuit::conjugate(const QuantumCircuit& other) const {
  if (num_qubits != other.num_qubits) {
    throw std::runtime_error("Mismatch in number of qubits in QuantumCircuit.conjugate.");
  }

  if (num_params() != 0 || other.num_params() != 0) {
    throw std::runtime_error("Unbound parameters, cannot performon QuantumCircuit.conjugate.");
  }

  QuantumCircuit qc(num_qubits);
  qc.append(other);
  qc.append(*this);
  qc.append(other.adjoint());
  return qc;
}

std::vector<QuantumCircuit> QuantumCircuit::split_into_unitary_components() const {
  std::vector<QuantumCircuit> components;

  if (instructions.size() == 0) {
    return components;
  }

  bool on_unitary_section = instruction_is_unitary(instructions[0]);
  QuantumCircuit qc(num_qubits);
  for (const auto& inst : instructions) {
    if (on_unitary_section) {
      if (instruction_is_unitary(inst)) {
        qc.add_instruction(inst);
      } else {
        components.push_back(qc);
        qc = QuantumCircuit(num_qubits);
        qc.add_instruction(inst);
        on_unitary_section = false;
      }
    } else {
      if (instruction_is_unitary(inst)) {
        components.push_back(qc);
        qc = QuantumCircuit(num_qubits);
        qc.add_instruction(inst);
        on_unitary_section = true;
      } else {
        qc.add_instruction(inst);
      }
    }
  }

  components.push_back(qc);

  return components;
}

Eigen::MatrixXcd QuantumCircuit::to_matrix(const std::optional<std::vector<double>>& params_opt) const {
  size_t nparams = num_params();
  if (params_opt) { 
    auto params = params_opt.value();
    if (params.size() < nparams) {
      throw std::invalid_argument("Unbound parameters; cannot convert circuit to matrix.");
    } else if (params.size() > nparams) {
      throw std::invalid_argument("Too many parameters passed; cannot convert circuit to matrix.");
    }

    QuantumCircuit qc = bind_params(params);
    return qc.to_matrix();
  } else {
    if (nparams > 0) {
      throw std::invalid_argument("Unbound parameters; cannot convert circuit to matrix.");
    }

    if (num_qubits > 15) {
      throw std::runtime_error("Cannot convert QuantumCircuit with n > 15 qubits to matrix.");
    }

    Eigen::MatrixXcd Q = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);

    uint32_t p = num_qubits;

    for (uint32_t i = 0; i < instructions.size(); i++) {
      if (instructions[i].control || instructions[i].target) {
        throw std::runtime_error("Cannot convert a classically-conditioned or targetted instruction to matrix.");
      }

			std::visit(quantumcircuit_utils::overloaded {
        [&Q, p](std::shared_ptr<Gate> gate) { 
          Q = embed_unitary(gate->define(), gate->qubits, p) * Q; 
        },
        [&Q, p](const FreeFermionGate& gate) { 
          auto g = gate.to_gate();
          Q = embed_unitary(g->define(), g->qubits, p) * Q;
        },
        [](const Measurement& m) { 
          throw std::invalid_argument("Cannot convert measurement to matrix."); 
        },
        [](const WeakMeasurement& m) { 
          throw std::invalid_argument("Cannot convert weak measurement to matrix."); 
        }
      }, instructions[i].inst);
    }

    return Q;
  }
}

// --- Library for building common circuits --- //

QuantumCircuit generate_haar_circuit(uint32_t num_qubits, uint32_t depth, bool pbc) {
  QuantumCircuit circuit(num_qubits);

  for (uint32_t i = 0; i < depth; i++) {
    for (uint32_t q = 0; q < num_qubits/2; q++) {
      auto [q1, q2] = get_targets(i, q, num_qubits);
      if (!pbc) {
        if (std::abs(int(q1) - int(q2)) > 1) {
          continue;
        }
      }

      circuit.add_gate(haar_unitary(2), {q1, q2});
    }
  }

  return circuit;
}

QuantumCircuit hardware_efficient_ansatz(
    uint32_t num_qubits, 
    uint32_t depth, 
    const std::vector<std::string>& rotation_gates,
    const std::string& entangling_gate,
    bool final_layer
  ) {

  QuantumCircuit circuit(num_qubits);

  for (uint32_t i = 0; i < depth; i++) {
    for (uint32_t q = 0; q < num_qubits/2; q++) {
      auto [q1, q2] = get_targets(i, q, num_qubits);

      for (auto const &s : rotation_gates) {
        auto gate = parse_gate(s, Qubits{q1});
        if (gate->num_qubits != 1) {
          throw std::runtime_error("Rotational gate must be one-qubit.");
        }
        circuit.add_gate(gate);
        gate = parse_gate(s, Qubits{q2});
        circuit.add_gate(gate);
      }

      auto entangler = parse_gate(entangling_gate, Qubits{q1, q2});
      if (entangler->num_qubits != 1) {
        throw std::runtime_error("Entangler gate must be two-qubit.");
      }
      circuit.add_gate(entangler);
    }
  }

  if (final_layer) {
    for (uint32_t q = 0; q < num_qubits; q++) {
      for (auto const &s : rotation_gates) {
        auto gate = parse_gate(s, Qubits{q});
        if (gate->num_qubits != 1) {
          throw std::runtime_error("Rotational gate must be one-qubit.");
        }
        circuit.add_gate(gate);
      }
    }
  }

  return circuit;
}

QuantumCircuit rotation_layer(uint32_t num_qubits, const std::optional<Qubits>& qubits_opt) {
  auto qubits = parse_qargs_opt(qubits_opt, num_qubits);

  QuantumCircuit circuit(num_qubits);

  for (auto const& q : qubits) {
    circuit.add_gate(std::make_shared<RxRotationGate>(Qubits{q}));
  }

  return circuit;
}

QuantumCircuit random_clifford(uint32_t num_qubits) {
  QuantumCircuit qc(num_qubits);

  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  random_clifford_impl(qubits, qc);

  return qc;
}
