#pragma once

#include <QuantumState.h>

#include "CliffordState.h"
#include "Tableau.h"
#include "TableauSIMD.h"

class QuantumCHPState : public CliffordState {
  private:
#ifdef __AVX__
    static constexpr bool avx_enabled = true;
#else
    static constexpr bool avx_enabled = false;
#endif

  public:
    using CliffordState::expectation;

    mutable std::unique_ptr<TableauBase> tableau;
    int print_mode;

    QuantumCHPState()=default;

    QuantumCHPState(uint32_t num_qubits, bool use_simd=avx_enabled);

    bool operator==(const QuantumCHPState& other) const {
      return (*tableau == *other.tableau);
    }

    bool const operator!=(QuantumCHPState& other) const {
      return !(*this == other);
    }

    virtual std::string to_string() const override;

    void rref();
    void xrref();

    void set_print_mode(const std::string& mode);

    Statevector to_statevector() const;

    virtual void h(uint32_t a) override;
    virtual void s(uint32_t a) override;
    virtual void sd(uint32_t a) override;

    virtual void cx(uint32_t a, uint32_t b) override;
    virtual void cy(uint32_t a, uint32_t b) override;
    virtual void cz(uint32_t a, uint32_t b) override;

    PauliString get_stabilizer(size_t i) const;
    PauliString get_destabilizer(size_t i) const;

    std::vector<PauliString> stabilizers() const;

    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const override;
		virtual std::vector<double> probabilities() const override;

    virtual void random_clifford(const Qubits& qubits) override;

    virtual double mzr_expectation(uint32_t a) const override;

    virtual MeasurementData mzr(uint32_t a, std::optional<bool> outcome=std::nullopt) override;

    virtual double sparsity() const override;

    virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    int xrank() const;
    int partial_xrank(const Qubits& qubits) const;

    int rank() const;
    int partial_rank(const Qubits& qubits) const;

    void set_x(size_t i, size_t j, bool v);
    void set_z(size_t i, size_t j, bool v);

    std::vector<char> serialize() const override;
    void deserialize(const std::vector<char>& bytes) override;
};
