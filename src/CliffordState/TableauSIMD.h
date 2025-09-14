#pragma once

#include <string>
#include <vector>
#include <random>
#include <variant>
#include <algorithm>

#include "QuantumState.h"
#include "QuantumCircuit.h"

#include "Tableau.h"

class TableauSIMD : public TableauBase {
  public:
    uint32_t width;
    std::vector<binary_word*> rows;

    // pwidth may be one longer
    uint32_t pwidth;
    binary_word* phase;

    TableauSIMD()=default;
    ~TableauSIMD();

    void set(size_t i, size_t j, binary_word v);
    bool get(size_t i, size_t j) const;
    void reset(int i);
    void rowsum(int i, int j);
    void swap(size_t i, size_t j);
    inline uint8_t get_xz(int i, int j) const {
      constexpr uint32_t num_paulis = binary_word_size()/2;
      uint32_t bit_ind = 2u*(j % num_paulis);
      return 0u | (((rows[i][j / num_paulis] >> bit_ind) & 3u) << 0u);
    }

    virtual uint8_t get_phase(size_t i) const override;

    TableauSIMD(uint32_t num_qubits);

    virtual Pauli get_pauli(size_t i, size_t j) const override;
    virtual PauliString get_stabilizer(size_t i) const override;
    virtual PauliString get_destabilizer(size_t i) const override;

    bool operator==(TableauSIMD& other);

    // Put tableau into reduced row echelon form
    virtual void rref(const Qubits& sites) override;
    virtual void rref() override;
    virtual uint32_t rank(const Qubits& sites) override;
    virtual uint32_t rank() override;
    virtual void xrref(const Qubits& sites) override;
    virtual void xrref() override;
    virtual uint32_t xrank(const Qubits& sites) override;
    virtual uint32_t xrank() override;

    //TableauSIMD partial_trace(const Qubits& qubits);

    virtual double bitstring_amplitude(const BitString& bits) override;

    virtual std::string to_string(bool print_destabilizers=true) const override;
    virtual std::string to_string_ops(bool print_destabilizers=true) const override;

    virtual void h(uint32_t a) override;
    virtual void s(uint32_t a) override;
    virtual void cx(uint32_t a, uint32_t b) override;

    // Returns a pair containing (1) wether the outcome of a measurement on qubit a is deterministic
    // and (2) the index on which the CHP algorithm performs rowsum if the mzr is random
    virtual std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) const override;

    virtual MeasurementData mzr(uint32_t a, std::optional<bool> outcome=std::nullopt) override;
};
