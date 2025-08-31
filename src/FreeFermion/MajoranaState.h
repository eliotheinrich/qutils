#pragma once

#include "FreeFermionState.h"

// Following https://arxiv.org/pdf/1112.2184
class MajoranaState : public FreeFermionState {
  private:
    Eigen::MatrixXd M;

    void particles_at(const Qubits& sites);

  public:
    MajoranaState(uint32_t L, std::optional<Qubits> sites);

    MajoranaState(const MajoranaState& other)
      : FreeFermionState(other), M(other.M) { }

    std::vector<double> williamson_eigenvalues(const Eigen::MatrixXd& A) const;

    Eigen::MatrixXd prepare_hamiltonian(const FreeFermionHamiltonian& H) const;

    void evolve(const Eigen::MatrixXd& U);

    virtual void evolve_hamiltonian(const FreeFermionHamiltonian& H, double t=1.0) override;

    void forced_majorana_measurement(size_t i, bool outcome);

    // According to https://arxiv.org/pdf/2210.05681
    virtual void forced_projective_measurement(size_t i, bool outcome) override;

    virtual double occupation(size_t i) const override;

    // TODO fix (broken)
    virtual double entanglement(const QubitSupport& support, uint32_t index) override;

    virtual std::string to_string() const override;
};

// Work in progress! Do not use!
class ExtendedMajoranaState : public FreeFermionState {
  private:
    uint32_t L;
    using State = std::pair<double, MajoranaState>;
    std::vector<State> states;


  public:
    ExtendedMajoranaState(uint32_t L, std::optional<Qubits> sites) : FreeFermionState(L) {
      states = {State{1.0, MajoranaState(L, sites)}};
    }

    virtual void evolve_hamiltonian(const FreeFermionHamiltonian& H, double t=1.0) override;

    virtual void forced_projective_measurement(size_t i, bool outcome) override;

    virtual std::string to_string() const override;

    virtual double occupation(size_t i) const override;

    void interaction(size_t i);

    virtual double entanglement(const QubitSupport& support, uint32_t index) override;

};
