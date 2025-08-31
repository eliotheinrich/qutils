#pragma once

#include <QuantumState.h>

#include "CliffordState.h"
#include "Tableau.h"

class QuantumCHPState : public CliffordState {
  public:
    mutable Tableau tableau;
    int print_mode;

    QuantumCHPState()=default;

    QuantumCHPState(uint32_t num_qubits)
      : CliffordState(num_qubits), tableau(Tableau(num_qubits)) {
    }

    bool operator==(QuantumCHPState& other) {
      return tableau == other.tableau;
    }

    bool operator!=(QuantumCHPState& other) {
      return !(tableau == other.tableau);
    }

    virtual std::string to_string() const override;

    void rref();

    void xrref();

    void set_print_mode(const std::string& mode);

    void rowsum(uint32_t q1, uint32_t q2);

    Statevector to_statevector() const;

    virtual void h(uint32_t a) override;

    virtual void s(uint32_t a) override;

    virtual void sd(uint32_t a) override;

    virtual void cx(uint32_t a, uint32_t b) override;

    virtual void cy(uint32_t a, uint32_t b) override;

    virtual void cz(uint32_t a, uint32_t b) override;

    PauliString get_row(size_t i) const;

    std::vector<PauliString> stabilizers() const;

    size_t size() const;

    virtual void random_clifford(const Qubits& qubits) override;

    virtual double mzr_expectation(uint32_t a) const override;

    virtual bool mzr(uint32_t a, std::optional<bool> outcome=std::nullopt) override;

    virtual double sparsity() const override;

    virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    int xrank();

    int partial_xrank(const Qubits& qubits);

    int rank();

    int partial_rank(const Qubits& qubits);

    void set_x(size_t i, size_t j, bool v);

    void set_z(size_t i, size_t j, bool v);

    std::vector<char> serialize() const override;
    void deserialize(const std::vector<char>& bytes) override;
};
