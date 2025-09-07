#pragma once

#include "QuantumState.h"

// Following https://arxiv.org/pdf/2307.12702
class GaussianState : public MagicQuantumState {
  private:
    Eigen::MatrixXcd amplitudes;

    void particles_at(const Qubits& sites);

    void single_particle();

    void all_particles();

    void checkerboard_particles();

  public:
    GaussianState()=default;
    GaussianState(uint32_t L, std::optional<Qubits> qubits=std::nullopt);

    void swap(size_t i, size_t j);

		virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    void orthogonalize();

    virtual void evolve(const FreeFermionGate& gate) override;

    virtual EvolveResult evolve(const QuantumCircuit& circuit, EvolveOpts opts=EvolveOpts()) override { 
      return QuantumState::evolve(circuit, opts);
    }

    virtual EvolveResult evolve(const QuantumCircuit& circuit, const Qubits& qubits, EvolveOpts opts=EvolveOpts()) override { 
      return QuantumState::evolve(circuit, qubits, opts);
    }

    virtual void evolve(const Eigen::MatrixXcd& gate, const Qubits& qubits) override {
      throw std::runtime_error("Cannot apply arbitrary gate to GaussianState.");
    }

    virtual MeasurementData mzr(uint32_t q, std::optional<bool> outcome=std::nullopt) override;
    virtual MeasurementData wmzr(uint32_t q, double beta, std::optional<bool> outcome=std::nullopt) override;

    double num_particles() const;
    double num_real_particles() const;

    Eigen::MatrixXcd covariance_matrix() const;
    Eigen::MatrixXcd majorana_covariance_matrix() const;
    std::complex<double> majorana_expectation(const std::vector<uint32_t>& indices) const;

    double occupation(size_t i) const;
    std::vector<double> occupation() const;

    virtual std::string to_string() const override;

    virtual std::complex<double> expectation(const PauliString& pauli) const override;

    virtual double expectation(const BitString& bits, std::optional<QubitSupport> support=std::nullopt) const override;
		virtual std::vector<double> probabilities() const override;

    virtual MeasurementData measure(const Measurement& m) override;
    virtual MeasurementData weak_measure(const WeakMeasurement& m) override;

    virtual double purity() const {
      return 1.0;
    }

    virtual std::shared_ptr<QuantumState> partial_trace(const Qubits& qubits) const {
      throw not_implemented();
    }

    struct glaze;
    virtual std::vector<char> serialize() const override;
    virtual void deserialize(const std::vector<char>& bytes) override;
};
