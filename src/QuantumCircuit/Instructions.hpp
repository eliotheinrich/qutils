#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <unordered_set>
#include <vector>
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
  constexpr std::complex<double> i_ = std::complex<double>(0.0, 1.0);

  struct H { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << sqrt2i_, sqrt2i_, sqrt2i_, -sqrt2i_).finished(); };

  struct I { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, 1.0).finished(); };
  struct X { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, 1.0, 1.0, 0.0).finished(); };
  struct Y { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << 0.0, -i_, i_, 0.0).finished(); };
  struct Z { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, -1.0).finished(); };

  struct sqrtX { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i_)/2.0, (1.0 - i_)/2.0, (1.0 - i_)/2.0, (1.0 + i_)/2.0).finished(); };
  struct sqrtY { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 + i_)/2.0, (-1.0 - i_)/2.0, (1.0 + i_)/2.0, (1.0 + i_)/2.0).finished(); };
  struct sqrtZ { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, i_).finished(); };

  struct sqrtXd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i_)/2.0, (1.0 + i_)/2.0, (1.0 + i_)/2.0, (1.0 - i_)/2.0).finished(); };
  struct sqrtYd { static inline const Eigen::Matrix2cd value = (Eigen::Matrix2cd() << (1.0 - i_)/2.0, (1.0 - i_)/2.0, (-1.0 + i_)/2.0, (1.0 - i_)/2.0).finished(); };
  struct sqrtZd { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, -i_).finished(); };

  struct T { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, sqrt2i_*(1.0 + i_)).finished(); };
  struct Td { static inline const Eigen::Vector2cd value = (Eigen::Vector2cd() << 1.0, sqrt2i_*(1.0 - i_)).finished(); };

  struct CX { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, 1, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, 1, 0, 0).finished(); };
  struct CY { static inline const Eigen::Matrix4cd value = (Eigen::Matrix4cd() << 1, 0, 0, 0, 
                                                                                  0, 0, 0, -i_, 
                                                                                  0, 0, 1, 0, 
                                                                                  0, i_, 0, 0).finished(); };
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
  double beta;
  std::optional<PauliString> pauli;
  std::optional<bool> outcome;

  WeakMeasurement(const Qubits& qubits, double beta, std::optional<PauliString> pauli=std::nullopt, std::optional<bool> outcome=std::nullopt);
  PauliString get_pauli() const;
  bool is_forced() const;
  bool get_outcome() const;
};

using Instruction = std::variant<std::shared_ptr<Gate>, Measurement, WeakMeasurement>;

template <>
struct fmt::formatter<Instruction> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const Instruction& inst, FormatContext& ctx) const {
    auto inst_to_string = [](const Instruction& inst) {
		  return std::visit(quantumcircuit_utils::overloaded {
          [](std::shared_ptr<Gate> gate) -> std::string {
            std::string gate_str = gate->label() + " ";
            for (auto const &q : gate->qubits) {
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
            std::string meas_str = fmt::format("weak_measure({}, {:.5f}) ", m.get_pauli(), m.beta);
            for (auto const &q : m.qubits) {
              meas_str += fmt::format("{} ", q);
            }

            if (m.outcome) {
              meas_str += fmt::format("-> {}", m.outcome.value());
            }
            return meas_str;
          }
      }, inst);
    };

    return fmt::format_to(ctx.out(), "{}", inst_to_string(inst));
  }
};

Instruction copy_instruction(const Instruction& inst);

Qubits get_instruction_support(const Instruction& inst);

bool instruction_is_unitary(const Instruction& inst);
