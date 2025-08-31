#pragma once

#include "FreeFermionState.h"

// Following https://arxiv.org/pdf/2307.12702
class GaussianState : public FreeFermionState {
  private:
    Eigen::MatrixXcd amplitudes;

    void particles_at(const Qubits& sites);

    void single_particle();

    void all_particles();

    void checkerboard_particles();

  public:
    GaussianState(uint32_t L, std::optional<Qubits> sites);

    void swap(size_t i, size_t j);

		virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    bool is_identity(const Eigen::MatrixXcd& A) const;

    void orthogonalize();
    bool check_orthogonality() const;
    void assert_ortho() const;

    Eigen::MatrixXcd prepare_hamiltonian(const FreeFermionHamiltonian& H) const;

    void evolve(const Eigen::MatrixXcd& U);
    virtual void evolve_hamiltonian(const FreeFermionHamiltonian& H, double t=1.0) override;

    void weak_measurement(const Eigen::MatrixXcd& U);
    void weak_measurement_hamiltonian(const FreeFermionHamiltonian& H, double beta=1.0);

    virtual void forced_projective_measurement(size_t i, bool outcome) override;

    virtual double num_particles() const override;
    double num_real_particles() const;

    Eigen::MatrixXcd covariance_matrix() const;

    virtual double occupation(size_t i) const override;

    virtual std::vector<double> occupation() const override;

    virtual std::string to_string() const override;
};
