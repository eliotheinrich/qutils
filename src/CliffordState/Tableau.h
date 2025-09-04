#pragma once

#include <string>
#include <vector>
#include <random>
#include <variant>
#include <algorithm>

#include "QuantumState.h"
#include "QuantumCircuit.h"

#include <Eigen/Dense>

#include <fmt/format.h>

class Tableau {
  public:
    uint32_t num_qubits;
    std::vector<PauliString> stabilizers;
    std::vector<PauliString> destabilizers;

    Tableau()=default;

    Tableau(uint32_t num_qubits);

    Tableau(uint32_t num_qubits, const std::vector<PauliString>& rows) : num_qubits(num_qubits), stabilizers(rows) {}

    uint32_t num_rows() const;

    Eigen::MatrixXi to_matrix() const;

    Statevector to_statevector() const;

    bool operator==(Tableau& other);

    // Put tableau into reduced row echelon form
    void rref(const Qubits& sites);
    void rref();
    uint32_t rank(const Qubits& sites);
    uint32_t rank();
    void xrref(const Qubits& sites);
    void xrref();
    uint32_t xrank(const Qubits& sites);
    uint32_t xrank();

    Tableau partial_trace(const Qubits& qubits);

    double bitstring_amplitude(const BitString& bits);

    inline void validate_qubit(uint32_t a) const {
      if (!(a >= 0 && a < num_qubits)) {
        std::string error_message = "A gate was applied to qubit " + std::to_string(a) + 
          ", which is outside of the allowed range (0, " + std::to_string(num_qubits) + ").";
        throw std::invalid_argument(error_message);
      }
    }

    std::string to_string(bool print_destabilizers=true) const;
    std::string to_string_ops(bool print_destabilizers=true) const;

    void h(uint32_t a);
    void s(uint32_t a);
    void sd(uint32_t a);
    void x(uint32_t a);
    void y(uint32_t a);
    void z(uint32_t a);
    void cx(uint32_t a, uint32_t b);
    void cz(uint32_t a, uint32_t b);

    // Returns a pair containing (1) wether the outcome of a measurement on qubit a is deterministic
    // and (2) the index on which the CHP algorithm performs rowsum if the mzr is random
    std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) const;

    MeasurementData mzr(uint32_t a, std::optional<bool> outcome=std::nullopt);

    double sparsity() const;
};
