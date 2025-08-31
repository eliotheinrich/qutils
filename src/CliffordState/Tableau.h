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
    bool track_destabilizers;
    uint32_t num_qubits;
    std::vector<PauliString> rows;

    Tableau()=default;

    Tableau(uint32_t num_qubits);

    Tableau(uint32_t num_qubits, const std::vector<PauliString>& rows) : track_destabilizers(false), num_qubits(num_qubits), rows(rows) {}

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

    inline void validate_qubit(uint32_t a) const {
      if (!(a >= 0 && a < num_qubits)) {
        std::string error_message = "A gate was applied to qubit " + std::to_string(a) + 
          ", which is outside of the allowed range (0, " + std::to_string(num_qubits) + ").";
        throw std::invalid_argument(error_message);
      }
    }

    std::string to_string(bool print_destabilizers=true) const;
    std::string to_string_ops(bool print_destabilizers=true) const;

    void rowsum(uint32_t h, uint32_t i);

    void evolve(const QuantumCircuit& qc);

		void evolve(const Instruction& inst);

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

    bool mzr(uint32_t a, std::optional<bool> outcome=std::nullopt);

    double sparsity() const;


    inline bool get_x(uint32_t i, uint32_t j) const { 
      return rows[i].get_x(j); 
    }

    inline bool get_z(uint32_t i, uint32_t j) const { 
      return rows[i].get_z(j); 
    }

    inline bool get_r(uint32_t i) const { 
      uint8_t r = rows[i].get_r();
      if (r == 0) {
        return false;
      } else if (r == 2) {
        return true;
      } else {
        throw std::runtime_error("Anomolous phase detected in Clifford tableau.");
      }
    }

    inline void set_x(uint32_t i, uint32_t j, bool v) { 
      rows[i].set_x(j, v); 
    }

    inline void set_z(uint32_t i, uint32_t j, bool v) { 
      rows[i].set_z(j, v); 
    }

    inline void set_r(uint32_t i, bool v) { 
      if (v) {
        rows[i].set_r(2);
      } else {
        rows[i].set_r(0);

      }
    }
};
