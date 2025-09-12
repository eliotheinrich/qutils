#pragma once

#include <string>
#include <vector>
#include <random>
#include <variant>
#include <algorithm>

#include "QuantumState.h"
#include "QuantumCircuit.h"

class TableauBase {
  public:
    uint32_t num_qubits;

    TableauBase()=default;
    TableauBase(uint32_t num_qubits) : num_qubits(num_qubits) {}

    bool operator==(const TableauBase& other) const;

    virtual Pauli get_pauli(size_t i, size_t j) const=0;
    virtual PauliString get_stabilizer(size_t i) const;
    virtual uint8_t get_phase(size_t i) const=0;

    virtual Eigen::MatrixXi to_matrix() const;

    virtual Statevector to_statevector() const;

    // Put tableau into reduced row echelon form
    virtual void rref(const Qubits& sites)=0;
    virtual void rref()=0;
    virtual uint32_t rank(const Qubits& sites)=0;
    virtual uint32_t rank()=0;
    virtual void xrref(const Qubits& sites)=0;
    virtual void xrref()=0;
    virtual uint32_t xrank(const Qubits& sites)=0;
    virtual uint32_t xrank()=0;

    virtual double bitstring_amplitude(const BitString& bits)=0;

    inline void validate_qubit(uint32_t a) const {
      if (!(a >= 0 && a < num_qubits)) {
        std::string error_message = "A gate was applied to qubit " + std::to_string(a) + 
          ", which is outside of the allowed range (0, " + std::to_string(num_qubits) + ").";
        throw std::invalid_argument(error_message);
      }
    }

    virtual std::string to_string(bool print_destabilizers=true) const=0;
    virtual std::string to_string_ops(bool print_destabilizers=true) const=0;

    virtual void h(uint32_t a)=0;
    virtual void s(uint32_t a)=0;
    virtual void sd(uint32_t a);
    virtual void x(uint32_t a);
    virtual void y(uint32_t a);
    virtual void z(uint32_t a);
    virtual void cx(uint32_t a, uint32_t b)=0;
    virtual void cz(uint32_t a, uint32_t b);

    // Returns a pair containing (1) wether the outcome of a measurement on qubit a is deterministic
    // and (2) the index on which the CHP algorithm performs rowsum if the mzr is random
    virtual std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) const=0;

    virtual MeasurementData mzr(uint32_t a, std::optional<bool> outcome)=0;

    virtual double sparsity() const;
};

class Tableau : public TableauBase {
  public:
    std::vector<PauliString> stabilizers;
    std::vector<PauliString> destabilizers;

    Tableau()=default;

    Tableau(uint32_t num_qubits);

    Tableau(uint32_t num_qubits, const std::vector<PauliString>& rows) : TableauBase(num_qubits), stabilizers(rows) {}

    virtual Pauli get_pauli(size_t i, size_t j) const override;
    virtual PauliString get_stabilizer(size_t i) const override;
    virtual uint8_t get_phase(size_t i) const override;

    bool operator==(Tableau& other);

    // Put tableau into reduced row echelon form
    virtual void rref(const Qubits& sites) override;
    virtual void rref() override;
    virtual uint32_t rank(const Qubits& sites) override;
    virtual uint32_t rank() override;
    virtual void xrref(const Qubits& sites) override;
    virtual void xrref() override;
    virtual uint32_t xrank(const Qubits& sites) override;
    virtual uint32_t xrank() override;

    Tableau partial_trace(const Qubits& qubits);

    virtual double bitstring_amplitude(const BitString& bits) override;

    virtual std::string to_string(bool print_destabilizers=true) const override;
    virtual std::string to_string_ops(bool print_destabilizers=true) const override;

    virtual void h(uint32_t a) override;
    virtual void s(uint32_t a) override;
    virtual void sd(uint32_t a) override;
    virtual void x(uint32_t a) override;
    virtual void y(uint32_t a) override;
    virtual void z(uint32_t a) override;
    virtual void cx(uint32_t a, uint32_t b) override;
    virtual void cz(uint32_t a, uint32_t b) override;

    // Returns a pair containing (1) wether the outcome of a measurement on qubit a is deterministic
    // and (2) the index on which the CHP algorithm performs rowsum if the mzr is random
    virtual std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) const override;

    virtual MeasurementData mzr(uint32_t a, std::optional<bool> outcome=std::nullopt) override;
};
