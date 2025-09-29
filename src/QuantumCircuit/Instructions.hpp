#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <unordered_set>
#include <vector>
#include <map>
#include <variant>
#include <complex>
#include <memory>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "PauliString.hpp"
#include "CircuitUtils.h"
#include "QuantumState/utils.hpp"

// --- Definitions for gates/measurements --- //

namespace gates {
  constexpr double sqrt2i_ = 0.707106781186547524400844362104849;
  constexpr std::complex<double> i = std::complex<double>(0.0, 1.0);

  struct H { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << sqrt2i_, sqrt2i_, sqrt2i_, -sqrt2i_).finished(); };

  struct I { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, 1.0).finished(); };
  struct X { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, 1.0, 1.0, 0.0).finished(); };
  struct Y { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0,-i, i, 0.0).finished(); };
  struct Z { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0,-1.0).finished(); };

  struct sqrtX { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i)/2.0, ( 1.0 - i)/2.0, (1.0 - i)/2.0, (1.0 + i)/2.0).finished(); };
  struct sqrtY { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i)/2.0, (-1.0 - i)/2.0, (1.0 + i)/2.0, (1.0 + i)/2.0).finished(); };
  struct sqrtZ { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, i).finished(); };

  struct sqrtXd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i)/2.0, (1.0 + i)/2.0, ( 1.0 + i)/2.0, (1.0 - i)/2.0).finished(); };
  struct sqrtYd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i)/2.0, (1.0 - i)/2.0, (-1.0 + i)/2.0, (1.0 - i)/2.0).finished(); };
  struct sqrtZd { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, -i).finished(); };

  struct T { static inline const Eigen::Vector2cd value  = (Eigen::Vector2cd() << 1.0, sqrt2i_*(1.0 + i)).finished(); };
  struct Td { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, sqrt2i_*(1.0 - i)).finished(); };

  struct CX { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, 1, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, 1, 0, 0).finished(); };
  struct CY { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0,-i, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, i, 0, 0).finished(); };
  struct CZ { static inline const Eigen::Vector4cd value = (Eigen::Vector4cd() << 1, 1, 1, -1).finished(); };
  struct SWAP { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished(); };
}

class Gate {
  public:
    Qubits qubits;
    uint32_t num_qubits;

    Gate(const Qubits& qubits)
      : qubits(qubits), num_qubits(qubits.size()) {
        if (!qargs_unique(qubits)) {
          throw std::runtime_error(fmt::format("Qubits {} provided to gate not unique.", qubits));
        }
      }

    virtual uint32_t num_params() const=0;

    virtual std::string label() const=0;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const=0;

    Eigen::MatrixXcd define() const {
      if (num_params() > 0) {
        throw std::invalid_argument("Unbound parameters; cannot define gate.");
      }

      return define(std::vector<double>());
    }

    virtual std::shared_ptr<Gate> adjoint() const=0;

    Eigen::MatrixXcd adjoint(const std::vector<double>& params) const {
      return adjoint()->define(params);
    }

    virtual bool is_clifford() const=0;

    virtual std::shared_ptr<Gate> clone()=0;
};

class SymbolicGate : public Gate {
  private:
    enum GateLabel {
      H, X, Y, Z, sqrtX, sqrtY, S, sqrtXd, sqrtYd, Sd, T, Td, CX, CY, CZ, SWAP
    };

    inline static std::unordered_set<SymbolicGate::GateLabel> clifford_gates = std::unordered_set<SymbolicGate::GateLabel>{
      SymbolicGate::GateLabel::H, 
      SymbolicGate::GateLabel::sqrtX, SymbolicGate::GateLabel::sqrtY, SymbolicGate::GateLabel::S, 
      SymbolicGate::GateLabel::sqrtXd, SymbolicGate::GateLabel::sqrtYd, SymbolicGate::GateLabel::Sd, 
      SymbolicGate::GateLabel::X, SymbolicGate::GateLabel::Y, SymbolicGate::GateLabel::Z, 
      SymbolicGate::GateLabel::CX, SymbolicGate::GateLabel::CY, SymbolicGate::GateLabel::CZ, 
      SymbolicGate::GateLabel::SWAP
    };

    inline static std::unordered_set<SymbolicGate::GateLabel> single_qubit_gates = std::unordered_set<SymbolicGate::GateLabel>{
      SymbolicGate::GateLabel::H, SymbolicGate::GateLabel::T, SymbolicGate::GateLabel::Td,
      SymbolicGate::GateLabel::sqrtX, SymbolicGate::GateLabel::sqrtY, SymbolicGate::GateLabel::S, 
      SymbolicGate::GateLabel::sqrtXd, SymbolicGate::GateLabel::sqrtYd, SymbolicGate::GateLabel::Sd, 
      SymbolicGate::GateLabel::X, SymbolicGate::GateLabel::Y, SymbolicGate::GateLabel::Z
    };

    inline static std::unordered_set<SymbolicGate::GateLabel> two_qubit_gates = std::unordered_set<SymbolicGate::GateLabel>{
      SymbolicGate::GateLabel::CX, SymbolicGate::GateLabel::CY, SymbolicGate::GateLabel::CZ, 
      SymbolicGate::GateLabel::SWAP
    };

    inline static std::unordered_map<SymbolicGate::GateLabel, SymbolicGate::GateLabel> adjoint_map = std::unordered_map<SymbolicGate::GateLabel, SymbolicGate::GateLabel>{
      {SymbolicGate::GateLabel::sqrtX, SymbolicGate::GateLabel::sqrtXd},
      {SymbolicGate::GateLabel::sqrtY, SymbolicGate::GateLabel::sqrtYd},
      {SymbolicGate::GateLabel::S, SymbolicGate::GateLabel::Sd},
      {SymbolicGate::GateLabel::sqrtXd, SymbolicGate::GateLabel::sqrtX},
      {SymbolicGate::GateLabel::sqrtYd, SymbolicGate::GateLabel::sqrtY},
      {SymbolicGate::GateLabel::Sd, SymbolicGate::GateLabel::S},
      {SymbolicGate::GateLabel::T, SymbolicGate::GateLabel::Td},
      {SymbolicGate::GateLabel::Td, SymbolicGate::GateLabel::T},
    };

    static bool str_equal_ci(const char* a, const char* b);

    static SymbolicGate::GateLabel parse_gate(const char* name);

    static const char* type_to_string(SymbolicGate::GateLabel g);

    static size_t num_qubits_for_gate(SymbolicGate::GateLabel g);

    static Eigen::MatrixXcd process_gate_data(const Eigen::MatrixXcd& data);

    static const std::unordered_map<SymbolicGate::GateLabel, Eigen::MatrixXcd> gate_map;

  public:
    SymbolicGate::GateLabel type;

    SymbolicGate(SymbolicGate::GateLabel type, const Qubits& qubits) : Gate(qubits), type(type) {
      if (num_qubits_for_gate(type) != qubits.size()) {
        throw std::runtime_error("Invalid qubits provided to SymbolicGate.");
      }
    }

    SymbolicGate(const char* name, const Qubits& qubits) : SymbolicGate(parse_gate(name), qubits) { }
    SymbolicGate(const std::string& name, const Qubits& qubits) : SymbolicGate(name.c_str(), qubits) { }

    virtual bool is_clifford() const override;

    virtual uint32_t num_params() const override;

    virtual std::string label() const override;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override;

    virtual std::shared_ptr<Gate> adjoint() const override;

    virtual std::shared_ptr<Gate> clone() override;
};

class MatrixGate : public Gate {
  public:
    Eigen::MatrixXcd data;
    std::string label_str;

    MatrixGate(const Eigen::MatrixXcd& data, const Qubits& qubits, const std::string& label_str)
      : Gate(qubits), data(data), label_str(label_str) {}

    MatrixGate(const Eigen::MatrixXcd& data, const Qubits& qubits)
      : MatrixGate(data, qubits, "U") {}


    virtual uint32_t num_params() const override;

    virtual std::string label() const override;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override;

    virtual std::shared_ptr<Gate> adjoint() const override;

    virtual bool is_clifford() const override;

    virtual std::shared_ptr<Gate> clone() override;
};

class PauliRotationGate : public Gate {
  private:
    bool adj;
    PauliString pauli;

  public:
    PauliRotationGate(const Qubits& qubits, const PauliString& pauli, bool adj=false) : Gate(qubits), pauli(pauli), adj(adj) {
      if (qubits.size() != pauli.num_qubits) {
        throw std::runtime_error(fmt::format("{} gate can only have {} qubits. Passed {}", label(), pauli.num_qubits, qubits));
      }

      if (!pauli.hermitian()) {
        throw std::runtime_error(fmt::format("Pauli {} provided to PauliRotationGate is not hermitian.", pauli));
      }
    }

    virtual uint32_t num_params() const override;

    virtual std::string label() const override;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override;

    virtual bool is_clifford() const override;

    virtual std::shared_ptr<Gate> adjoint() const override;

    virtual std::shared_ptr<Gate> clone() override;
};

class RxRotationGate : public Gate {
  private:
    bool adj;

  public:
    RxRotationGate(const Qubits& qubits, bool adj=false) : Gate(qubits), adj(adj) {
      if (qubits.size() != 1) {
        std::string error_message = "Rx gate can only have a single qubit. Passed " + std::to_string(qubits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override;

    virtual std::string label() const override;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override;

    virtual bool is_clifford() const override;

    virtual std::shared_ptr<Gate> adjoint() const override;

    virtual std::shared_ptr<Gate> clone() override;
};

class RyRotationGate : public Gate {
  private:
    bool adj;

  public:
    RyRotationGate(const Qubits& qubits, bool adj=false) : Gate(qubits), adj(adj) {
      if (qubits.size() != 1) {
        std::string error_message = "Ry gate can only have a single qubit. Passed " + std::to_string(qubits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override;
    
    virtual std::string label() const override;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override;

    virtual bool is_clifford() const override;

    virtual std::shared_ptr<Gate> adjoint() const override;

    virtual std::shared_ptr<Gate> clone() override;
};

class RzRotationGate : public Gate {
  private:
    bool adj;

  public:
    RzRotationGate(const Qubits& qubits, bool adj=false) : Gate(qubits), adj(adj) {
      if (qubits.size() != 1) {
        std::string error_message = "Rz gate can only have a single qubit. Passed " + std::to_string(qubits.size()) + ".";
        throw std::invalid_argument(error_message);
      }
    }

    virtual uint32_t num_params() const override;

    virtual std::string label() const override;

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override;

    virtual bool is_clifford() const override;

    virtual std::shared_ptr<Gate> adjoint() const override;

    virtual std::shared_ptr<Gate> clone() override;
};

template <class GateType>
class MemoizedGate : public GateType {
  private:
    bool adj;

    static std::vector<Eigen::MatrixXcd> memoized_gates;
    static bool defined;

    static void generate_memoized_gates(uint32_t res, double min, double max) {
      MemoizedGate<GateType>::memoized_gates = std::vector<Eigen::MatrixXcd>(res);
      double bin_width = (max - min)/res;
      Qubits qubits{0};

      for (uint32_t i = 0; i < res; i++) {
        double d = min + bin_width*i;

        GateType gate(qubits);
        std::vector<double> params{d};
        MemoizedGate<GateType>::memoized_gates[i] = gate.define(params);
      }

      MemoizedGate<GateType>::defined = true;
    }

    static uint32_t get_idx(double d, uint32_t res, double min, double max) {
      double dt = std::fmod(d, 2*M_PI);

      double bin_width = static_cast<double>(max - min)/res;
      return static_cast<uint32_t>((dt - min)/bin_width);
    }

  public:
    MemoizedGate(const Qubits& qubits, bool adj=false) : GateType(qubits), adj(adj) {}

    virtual Eigen::MatrixXcd define(const std::vector<double>& params) const override {
      if (!MemoizedGate<GateType>::defined) {
        MemoizedGate<GateType>::generate_memoized_gates(200, 0, 2*M_PI);
      }

      if (params.size() != this->num_params()) {
        std::string error_message = "Invalid number of params passed to define(). Expected " 
                                   + std::to_string(this->num_params()) + ", received " + std::to_string(params.size()) + ".";
        throw std::invalid_argument(error_message);
      }

      double d = params[0];
      uint32_t idx = MemoizedGate<GateType>::get_idx(d, 200, 0, 2*M_PI);

      Eigen::MatrixXcd g = MemoizedGate<GateType>::memoized_gates[idx];

      if (adj) {
        g = g.adjoint();
      }

      return g;
    }

    virtual bool is_clifford() const override {
      return false;
    }
    
    virtual std::shared_ptr<Gate> adjoint() const override {
      return std::shared_ptr<Gate>(new MemoizedGate<GateType>(this->qubits, !adj));
    }

    virtual std::shared_ptr<Gate> clone() override {
      return std::shared_ptr<Gate>(new MemoizedGate<GateType>(this->qubits, adj));
    }
};

template <class GateType>
std::vector<Eigen::MatrixXcd> MemoizedGate<GateType>::memoized_gates;

template <class GateType>
bool MemoizedGate<GateType>::defined = false;

std::shared_ptr<Gate> parse_gate(const std::string& s, const Qubits& qubits);

static inline bool is_hermitian(const Eigen::MatrixXcd& H) {
  return H.isApprox(H.adjoint());
}

static inline bool is_antisymmetric(const Eigen::MatrixXcd& A) {
  return A.isApprox(-A.transpose());
}

static inline bool is_unitary(const Eigen::MatrixXcd& U) {
  Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(U.rows(), U.cols());
  return (U.adjoint() * U).isApprox(I);
}

struct QuadraticFermionTerm {
  uint32_t i;
  uint32_t j;
  double a;
  bool adj;
};

Eigen::MatrixXcd term_to_matrix(const QuadraticFermionTerm& term);
QubitInterval get_term_support(const QuadraticFermionTerm& term);
Eigen::MatrixXcd fermion_operator(size_t k, size_t num_qubits);
PauliString majorana_operator(size_t k, size_t num_qubits);

class FreeFermionGate {
  private:
    uint32_t num_qubits;
    std::vector<QuadraticFermionTerm> terms;

  public:
    std::optional<double> t;
    bool adj;

    FreeFermionGate()=default;
    FreeFermionGate(const FreeFermionGate& other)=default;
    
    FreeFermionGate(uint32_t num_qubits, std::optional<double> t=std::nullopt, bool adj=false)
      : num_qubits(num_qubits), t(t), adj(adj) { }


    uint32_t num_params() const {
      return t ? 0 : 1;
    }

    void add_term(uint32_t i, uint32_t j, double a, bool adj=true) {
      size_t i1 = std::min(i, j);
      size_t i2 = std::max(i, j);
      double sign = (i1 != i && !adj) ? -1.0 : 1.0;
      double amplitude = sign * a;
      terms.push_back({i, j, amplitude, adj});
    }

    std::string label() const;

    std::string to_string() const;

    Qubits get_support() const;
    Eigen::MatrixXcd to_matrix() const;
    std::shared_ptr<Gate> to_gate() const;

    FreeFermionGate bind_parameters(const std::vector<double>& params) const;

    void apply_qubit_map(const Qubits& qubits);

    Eigen::MatrixXcd to_hamiltonian() const;

    FreeFermionGate adjoint() const {
      auto gate = FreeFermionGate(*this);
      gate.adj = !gate.adj;
      return gate;
    }

    bool is_clifford() const {
      return false;
    }
};

// Gates of the form e^{it(a1*P1 + a2*P2 + ...)} 
// where P1...PN are Pauli strings, a1...aN are real numbers, and [Pi, Pj] = 0
struct PauliTerm {
  double a;
  PauliString pauli;
  Qubits support;
};
class CommutingHamiltonianGate {
  private:
    uint32_t num_qubits;

  public:
    std::vector<PauliTerm> terms;
    std::optional<double> t;
    bool adj;

    CommutingHamiltonianGate()=default;
    CommutingHamiltonianGate(const CommutingHamiltonianGate& other)=default;
    
    CommutingHamiltonianGate(uint32_t num_qubits, std::optional<double> t=std::nullopt, bool adj=false)
      : num_qubits(num_qubits), t(t), adj(adj) { }


    uint32_t num_params() const {
      return t ? 0 : 1;
    }

    void add_term(double a, const PauliString& p, const Qubits& qubits) {
      if (p.num_qubits != qubits.size()) {
        throw std::runtime_error(fmt::format("Invalid number of qubits passed to CommutingHamiltonianGate.add_term()."));
      }

      terms.push_back({a, p, qubits});
    }

    std::string label() const;

    std::string to_string() const;

    Qubits get_support() const;
    Eigen::MatrixXcd to_matrix() const;
    std::shared_ptr<Gate> to_gate() const;

    CommutingHamiltonianGate bind_parameters(const std::vector<double>& params) const;

    void apply_qubit_map(const Qubits& qubits);

    Eigen::MatrixXcd to_hamiltonian() const;

    CommutingHamiltonianGate adjoint() const {
      auto gate = CommutingHamiltonianGate(*this);
      gate.adj = !gate.adj;
      return gate;
    }

    bool is_clifford() const {
      return false;
    }
};

struct Measurement {
  Qubits qubits;
  std::optional<PauliString> pauli;
  std::optional<bool> outcome;

  Measurement(const Qubits& qubits, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);
  static Measurement computational_basis(uint32_t q, std::optional<bool> outcome=std::nullopt);
  PauliString get_pauli() const;
  bool is_basis() const;
  bool is_forced() const;
  bool get_outcome() const;
};

struct WeakMeasurement {
  Qubits qubits;
  std::optional<double> beta;
  std::optional<PauliString> pauli;
  std::optional<bool> outcome;

  WeakMeasurement(const Qubits& qubits, std::optional<double> beta=std::nullopt, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);
  size_t num_params() const;
  WeakMeasurement bind_parameters(const std::vector<double>& beta) const;
  PauliString get_pauli() const;
  bool is_basis() const;
  bool is_forced() const;
  bool get_outcome() const;
};

using QuantumInstruction = std::variant<
  std::shared_ptr<Gate>, 
  FreeFermionGate, 
  CommutingHamiltonianGate, 
  Measurement, 
  WeakMeasurement
>;

struct ClassicalInstruction {
  enum class OpType { NOT, AND, OR, XOR, NAND, CLEAR };
  OpType op;
  std::vector<uint32_t> bits;

  void apply(BitString& target) const {
    switch (op) {
      case OpType::NOT: {
        bool b = target.get(bits[0]);
        target.set(bits[1], !b);
        break;
      } case OpType::AND: {
        bool b1 = target.get(bits[0]);
        bool b2 = target.get(bits[1]);
        target.set(bits[2], b1 && b2);
        break;
      } case OpType::OR: {
        bool b1 = target.get(bits[0]);
        bool b2 = target.get(bits[1]);
        target.set(bits[2], b1 || b2);
        break;
      } case OpType::XOR: {
        bool b1 = target.get(bits[0]);
        bool b2 = target.get(bits[1]);
        target.set(bits[2], b1 ^ b2);
        break;
      } case OpType::NAND: {
        bool b1 = target.get(bits[0]);
        bool b2 = target.get(bits[1]);
        target.set(bits[2], !(b1 && b2));
        break;
      } case OpType::CLEAR: {
        target.set(bits[0], 0);
      }
    }
  }

  std::string to_string() const {
    switch (op) {
      case (OpType::NOT): {
        return fmt::format("NOT {} {}", bits[0], bits[1]);
      } case (OpType::AND): {
        return fmt::format("AND {} {} {}", bits[0], bits[1], bits[2]);
      } case (OpType::OR): {
        return fmt::format("OR {} {} {}", bits[0], bits[1], bits[2]);
      } case (OpType::XOR): {
        return fmt::format("OR {} {} {}", bits[0], bits[1], bits[2]);
      } case (OpType::NAND): {
        return fmt::format("NAND {} {} {}", bits[0], bits[1], bits[2]);
      } case (OpType::CLEAR): {
        return fmt::format("CLEAR {}", bits[0]);
      }
    }
  }
};

using TargetOpt = std::optional<uint32_t>;
using ControlOpt = std::optional<uint32_t>;

struct ConditionedInstruction {
  ConditionedInstruction()=default;
  ConditionedInstruction(const QuantumInstruction& inst) : inst(inst) { }
  ConditionedInstruction(const QuantumInstruction& inst, ControlOpt control, TargetOpt target)
    : inst(inst), control(control), target(target) { }

  QuantumInstruction inst;
  ControlOpt control;
  TargetOpt target;

  bool should_execute(const BitString& bits) const {
    return control ? bits.get(control.value()) : true;
  }
};

using Instruction = std::variant<QuantumInstruction, ClassicalInstruction, ConditionedInstruction>;

template <>
struct fmt::formatter<QuantumInstruction> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const QuantumInstruction& inst, FormatContext& ctx) const {
    auto inst_to_string = [](const QuantumInstruction& qinst) {
		  return std::visit(quantumcircuit_utils::overloaded {
        [](std::shared_ptr<Gate> gate) -> std::string {
          std::string gate_str = gate->label() + " ";
          for (auto const &q : gate->qubits) {
            gate_str += fmt::format("{} ", q);
          }

          return gate_str;
        },
        [](const FreeFermionGate& gate) -> std::string {
          std::string gate_str = gate.label() + " ";
          for (auto const &q : gate.get_support()) {
            gate_str += fmt::format("{} ", q);
          }

          return gate_str;
        },
        [](const CommutingHamiltonianGate& gate) -> std::string {
          std::string gate_str = gate.label() + " ";
          for (auto const &q : gate.get_support()) {
            gate_str += fmt::format("{} ", q);
          }

          return gate_str;
        },
        [](const Measurement& m) -> std::string {
          if (m.is_basis()) {
            return fmt::format("mzr {}{}", m.qubits[0], m.is_forced() ? fmt::format(" -> {}", m.get_outcome()) : "");
          }
          std::string meas_str = fmt::format("measure({}) ", m.get_pauli());
          for (auto const &q : m.qubits) {
            meas_str += fmt::format("{} ", q);
          }

          if (m.outcome) {
            meas_str += fmt::format("-> {}", m.outcome.value());
          }
          return meas_str;
        },
        [](const WeakMeasurement& m) -> std::string {
          std::string beta = m.beta ? fmt::format("{:.5f}", m.beta.value()) : "beta";
          std::string meas_str = fmt::format("weak_measure({}, {}) ", beta, m.get_pauli());
          for (auto const &q : m.qubits) {
            meas_str += fmt::format("{} ", q);
          }

          if (m.outcome) {
            meas_str += fmt::format("-> {}", m.outcome.value());
          }
          return meas_str;
        }
      }, qinst);
    };

    return fmt::format_to(ctx.out(), "{}", inst_to_string(inst));
  }
};

template <>
struct fmt::formatter<ClassicalInstruction> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const ClassicalInstruction& clinst, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}", clinst.to_string());
  }
};

template <>
struct fmt::formatter<ConditionedInstruction> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const ConditionedInstruction& cinst, FormatContext& ctx) const {
    std::string s = fmt::format("{}", cinst.inst);
    if (cinst.control) {
      s += fmt::format(" c-{}", cinst.control.value());
    }

    if (cinst.target) {
      s += fmt::format(" t-{}", cinst.target.value());
    }

    return fmt::format_to(ctx.out(), "{}", s);
  }
};


template <>
struct fmt::formatter<Instruction> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const Instruction& inst, FormatContext& ctx) const {
    auto inst_to_string = [](const Instruction& inst) {
      return std::visit(quantumcircuit_utils::overloaded {
        [](const QuantumInstruction& qinst) {
          return fmt::format("{}", qinst);
        },
        [](const ClassicalInstruction& clinst) {
          return fmt::format("{}", clinst);
        },
        [](const ConditionedInstruction& cinst) {
          return fmt::format("{}", cinst);
        }
      }, inst);
    };

    return fmt::format_to(ctx.out(), "{}", inst_to_string(inst));
  }
};

Instruction copy_instruction(const Instruction& inst);
size_t get_instruction_num_params(const Instruction& inst);
Qubits get_instruction_support(const Instruction& inst);
Qubits get_instruction_classical_support(const Instruction& inst);
bool instruction_is_unitary(const Instruction& inst);
Instruction instruction_adjoint(const Instruction& inst);
bool instruction_is_classical(const Instruction& inst);
bool instruction_is_quantum(const Instruction& inst);
