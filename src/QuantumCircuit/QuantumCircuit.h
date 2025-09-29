#pragma once

#include <random>
#include "CircuitUtils.h"
#include "Graph.hpp"

#include "Instructions.hpp"

#include <iostream>

#include <fmt/format.h>

// --- Definitions for QuantumCircuit --- //

using CircuitDAG = DirectedGraph<Instruction>;

class QuantumCircuit {
  private:
    uint32_t num_qubits;
    uint32_t num_cbits;

    // Maps index of measurement to index of corresponding instruction
    // i.e. measurement_map[0] contains the index of the first measurement, etc.
    std::vector<size_t> measurement_map;

    // Maps index of parameter to index of corresponding instruction
    // i.e. parameter_map[0] contains the index of the instruction containing the first parameter, etc.
    std::vector<size_t> parameter_map;

    void validate_instruction(const Instruction& inst) const;

    template <typename GateType, typename...Args>
    void add_gate(const Qubits& qubits, std::optional<std::vector<double>> theta_opt=std::nullopt, ControlOpt control=std::nullopt, const Args&... args) {
      if (theta_opt) {
        GateType gate(qubits, args...);
        add_gate(std::make_shared<MatrixGate>(gate.define(theta_opt.value()), qubits, gate.label()), control);
      } else {
        add_gate(std::make_shared<GateType>(qubits, args...), control);
      }
    }

    static inline std::optional<std::vector<double>> to_vector(std::optional<double> theta_opt) {
      if (theta_opt) {
        return std::vector<double>{theta_opt.value()};
      } else {
        return std::nullopt;
      }
    }

  public:
    std::vector<Instruction> instructions;

    QuantumCircuit() : num_qubits(0), num_cbits(0) {}

    QuantumCircuit(uint32_t num_qubits, uint32_t num_cbits=0) : num_qubits(num_qubits), num_cbits(num_cbits) {}

    QuantumCircuit(const QuantumCircuit& qc) : num_qubits(qc.num_qubits), num_cbits(qc.num_cbits) { 
      append(qc); 
      measurement_map = qc.measurement_map;
      parameter_map = qc.parameter_map;
    };

    CircuitDAG to_dag() const;
    static QuantumCircuit to_circuit(const CircuitDAG& dag, uint32_t num_qubits, uint32_t num_cbits, const std::vector<size_t>& measurement_map, const std::vector<size_t>& parameter_map, bool ltr=true);
    QuantumCircuit simplify(bool ltr) const;

    uint32_t get_num_qubits() const {
      return num_qubits;
    }

    uint32_t get_num_cbits() const {
      return num_cbits;
    }

    void resize_qubits(uint32_t num_qubits) {
      this->num_qubits = num_qubits;
    }

    void resize_cbits(uint32_t num_cbits) {
      this->num_cbits = num_cbits;
    }

    void apply_qubit_map(const Qubits& qubits);
    void apply_cbit_map(const Qubits& bits);

    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& stream, const QuantumCircuit& qc) {
      stream << qc.to_string();
      return stream;
    }

    uint32_t length() const;

    bool is_unitary() const;
    bool is_clifford() const;

    template <typename... QuantumStates>
    void apply(QuantumStates&... states) const {
      ([&] {
       if constexpr (std::is_same_v<std::decay_t<QuantumStates>, QuantumCircuit>) {
         states.append(*this);
       } else {
         states.evolve(*this);
       }
      }(), ...);
    }

    template <typename... QuantumStates>
    void apply(const Qubits& qubits, QuantumStates&... states) const {
      ([&] {
       if constexpr (std::is_same_v<std::decay_t<QuantumStates>, QuantumCircuit>) {
         states.append(*this, qubits);
       } else {
         states.evolve(*this, qubits);
       }
      }(), ...);
    }

    Qubits get_support() const;

    void add_instruction(const Instruction& inst);
    void add_controlled_instruction(const QuantumInstruction& qinst, size_t control);
    void add_targeted_instruction(const QuantumInstruction& qinst, size_t target);

    void add_measurement(const Measurement& m, TargetOpt target=std::nullopt);
    void add_measurement(const Qubits& qubits, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt, TargetOpt target=std::nullopt) {
      Measurement m(qubits, pauli, outcome);
      add_measurement(m, target);
    }

    void mzr(uint32_t q, TargetOpt target=std::nullopt) { 
      Measurement m({q}, std::nullopt, std::nullopt);
      add_measurement(m, target); 
    }
    void add_weak_measurement(const WeakMeasurement& m, TargetOpt target=std::nullopt);
    void add_weak_measurement(const Qubits& qubits, double beta, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt, TargetOpt target=std::nullopt) {
      WeakMeasurement m(qubits, beta, pauli, outcome);
      add_weak_measurement(m, target);
    }
    void wmzr(uint32_t q, std::optional<double> beta=std::nullopt, TargetOpt target=std::nullopt) { 
      WeakMeasurement m({q}, beta, std::nullopt, std::nullopt);
      add_weak_measurement(m, target); 
    }

    void add_gate(const FreeFermionGate& gate, ControlOpt control=std::nullopt);
    void add_gate(const CommutingHamiltonianGate& gate, ControlOpt control=std::nullopt);
    void add_gate(const std::shared_ptr<Gate> &gate, ControlOpt control=std::nullopt);
    void add_gate(const Eigen::MatrixXcd& gate, const Qubits& qubits, ControlOpt control=std::nullopt);
    void add_gate(const Eigen::Matrix2cd& gate, uint32_t qubit, ControlOpt control=std::nullopt);
    void add_gate(const std::string& name, const Qubits& qubits, ControlOpt control=std::nullopt);

    void h(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("h", {q}, control);
    }

    void s(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("s", {q}, control);
    }

    void sd(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sd", {q}, control);
    }

    void t(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("t", {q}, control);
    }

    void td(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("td", {q}, control);
    }

    void x(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("x", {q}, control);
    }

    void y(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("y", {q}, control);
    }

    void z(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("z", {q}, control);
    }

    void sqrtX(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sqrtX", {q}, control);
    }

    void sqrtY(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sqrtY", {q}, control);
    }

    void sqrtZ(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sqrtZ", {q}, control);
    }

    void sqrtXd(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sqrtXd", {q}, control);
    }

    void sqrtYd(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sqrtYd", {q}, control);
    }

    void sqrtZd(uint32_t q, ControlOpt control=std::nullopt) {
      add_gate("sqrtZd", {q}, control);
    }

    void cx(uint32_t q1, uint32_t q2, ControlOpt control=std::nullopt) {
      add_gate("cx", {q1, q2}, control);
    }

    void cy(uint32_t q1, uint32_t q2, ControlOpt control=std::nullopt) {
      add_gate("cy", {q1, q2}, control);
    }

    void cz(uint32_t q1, uint32_t q2, ControlOpt control=std::nullopt) {
      add_gate("cz", {q1, q2}, control);
    }

    void swap(uint32_t q1, uint32_t q2, ControlOpt control=std::nullopt) {
      add_gate("swap", {q1, q2}, control);
    }

    void rx(uint32_t q, std::optional<double> theta_opt=std::nullopt, ControlOpt control=std::nullopt) {
      add_gate<RxRotationGate>({q}, to_vector(theta_opt), control);
    }

    void ry(uint32_t q, std::optional<double> theta_opt=std::nullopt, ControlOpt control=std::nullopt) {
      add_gate<RyRotationGate>({q}, to_vector(theta_opt), control);
    }

    void rz(uint32_t q, std::optional<double> theta_opt=std::nullopt, ControlOpt control=std::nullopt) {
      add_gate<RzRotationGate>({q}, to_vector(theta_opt), control);
    }

    void rp(const Qubits& qubits, const PauliString& pauli, std::optional<double> theta_opt=std::nullopt, ControlOpt control=std::nullopt) {
      add_gate<PauliRotationGate>(qubits, to_vector(theta_opt), control, pauli);
    }

    void random_clifford(const Qubits& qubits);

    void cl_not(uint32_t control, uint32_t target);
    void cl_and(uint32_t control1, uint32_t control2, uint32_t target);
    void cl_or(uint32_t control1, uint32_t control2, uint32_t target);
    void cl_xor(uint32_t control1, uint32_t control2, uint32_t target);
    void cl_nand(uint32_t control1, uint32_t control2, uint32_t target);
    void cl_clear(uint32_t target);

    void append(const QuantumCircuit& other);
    void append(const QuantumCircuit& other, const Qubits& qubits);
    void append(const Instruction& inst, ControlOpt control=std::nullopt);

    void erase(size_t i);
    void insert(size_t i, const QuantumInstruction& qinst);

    QuantumCircuit bind_parameters(const std::vector<double>& params) const;
    QuantumCircuit bind_measurement_outcomes(const std::vector<bool>& outcomes) const;
    size_t get_num_measurements() const;
    std::vector<size_t> get_measurement_map() const {
      return measurement_map;
    }
    void set_measurement_map(const std::vector<size_t>& map) {
      measurement_map = map;
    }
    std::variant<Measurement, WeakMeasurement> get_measurement(size_t i) const;

    size_t get_num_parameters() const;
    std::vector<size_t> get_parameter_map() const {
      return parameter_map;
    }
    void set_parameter_map(const std::vector<size_t>& map) {
      parameter_map = map;
    }

    bool instruction_is_measurement(size_t i) const;

    QuantumCircuit adjoint(const std::optional<std::vector<double>>& params_opt=std::nullopt) const;
    QuantumCircuit reverse() const;
    QuantumCircuit conjugate(const QuantumCircuit& other) const;

    std::vector<QuantumCircuit> split_into_unitary_components() const;

    Eigen::MatrixXcd to_matrix(const std::optional<std::vector<double>>& params_opt=std::nullopt) const;
};

template <>
struct fmt::formatter<QuantumCircuit> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const QuantumCircuit& qc, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", qc.to_string());
  }
};


// --- Building common circuits --- //
QuantumCircuit generate_haar_circuit(uint32_t num_qubits, uint32_t depth, bool pbc=true);
QuantumCircuit hardware_efficient_ansatz(uint32_t num_qubits, uint32_t depth, const std::vector<std::string>& rotation_gates, const std::string& entangling_gate = "cz", bool final_layer = true);
QuantumCircuit rotation_layer(uint32_t num_qubits, const std::optional<Qubits>& qargs_opt = std::nullopt);
QuantumCircuit random_clifford(uint32_t num_qubits);
