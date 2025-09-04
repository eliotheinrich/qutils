#pragma once

#include "QuantumState.h"
#include "QuantumCircuit.h"
#include "Random.hpp"

#include <algorithm>

enum CliffordType { CHP, GraphSim };

class CliffordState : public QuantumState {
  public:
    size_t num_qubits;
    CliffordState()=default;

    CliffordState(uint32_t num_qubits) : QuantumState(num_qubits), num_qubits(num_qubits) {}
    virtual ~CliffordState() {}

    virtual EvolveResult evolve(const QuantumCircuit& qc, const Qubits& qubits, EvolveOpts opts=EvolveOpts()) override;
    virtual EvolveResult evolve(const QuantumCircuit& qc, EvolveOpts opts=EvolveOpts()) override;
		virtual std::optional<MeasurementData> evolve(const Instruction& inst) override;
    virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override;

    virtual void h(uint32_t a)=0;
    virtual void s(uint32_t a)=0;
    virtual void sd(uint32_t a);
    virtual void x(uint32_t a);
    virtual void y(uint32_t a);
    virtual void z(uint32_t a);
    virtual void sqrtx(uint32_t a);
    virtual void sqrty(uint32_t a);
    virtual void sqrtz(uint32_t a);
    virtual void sqrtxd(uint32_t a);
    virtual void sqrtyd(uint32_t a);
    virtual void sqrtzd(uint32_t a);
    virtual void cz(uint32_t a, uint32_t b)=0;
    virtual void cx(uint32_t a, uint32_t b);
    virtual void cy(uint32_t a, uint32_t b);
    virtual void swap(uint32_t a, uint32_t b);

    virtual double mzr_expectation(uint32_t a) const=0;
    virtual double mzr_expectation();

    virtual double mxr_expectation(uint32_t a);
    virtual double mxr_expectation();

    virtual double myr_expectation(uint32_t a);
    virtual double myr_expectation();

    virtual MeasurementData mzr(uint32_t a, std::optional<bool> outcome=std::nullopt)=0;
    virtual MeasurementData mxr(uint32_t a, std::optional<bool> outcome=std::nullopt);
    virtual MeasurementData myr(uint32_t a, std::optional<bool> outcome=std::nullopt);

    virtual double sparsity() const=0;

    virtual MeasurementData measure(const Measurement& m) override;
    virtual MeasurementData weak_measure(const WeakMeasurement& m) override;

    // NOTE: the sign of this is not guaranteed, since generic Clifford states do not track global phase
    virtual std::complex<double> expectation(const PauliString& pauli) const override;

    virtual double purity() const override;

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const override;
};
