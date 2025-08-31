#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

#include <EntanglementEntropyState.hpp>

static inline bool is_hermitian(const Eigen::MatrixXcd& H) {
  return H.isApprox(H.adjoint());
}

static inline bool is_antisymmetric(const Eigen::MatrixXcd& A) {
  return A.isApprox(-A.transpose());
}

static inline bool is_unitary(const Eigen::MatrixXcd& U) {
  Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(U.rows(), U.cols());
  return (U.adjoint() * U).isApprox(I);
}

struct FreeFermionHamiltonian {
  struct QuadraticTerm {
    uint32_t i;
    uint32_t j;
    double a;
  };

  uint32_t L;
  std::vector<QuadraticTerm> terms;
  std::vector<QuadraticTerm> nc_terms;

  FreeFermionHamiltonian(uint32_t L) : L(L) {
    terms = {};
    nc_terms = {};
  }

  void add_term(uint32_t i, uint32_t j, double a) {
    terms.push_back({i, j, a});
  }

  void add_nonconserving_term(uint32_t i, uint32_t j, double a) {
    nc_terms.push_back({i, j, a});
  }
};

class FreeFermionState : public EntanglementEntropyState {
  protected:
    uint32_t L;

  public:
    FreeFermionState(uint32_t L) : EntanglementEntropyState(L), L(L) { }

    size_t system_size() const {
      return L;
    }

    virtual void evolve_hamiltonian(const FreeFermionHamiltonian& H, double t=1.0)=0;
    virtual void forced_projective_measurement(size_t i, bool outcome)=0;
    virtual bool projective_measurement(size_t i);

    virtual double num_particles() const;

    virtual double occupation(size_t i) const=0;

    virtual std::vector<double> occupation() const;

    virtual std::string to_string() const=0;
};


class AmplitudeFermionState : public FreeFermionState {
  public:
    Eigen::MatrixXcd amplitudes;

    AmplitudeFermionState(uint32_t L) : FreeFermionState(L) {
      particles_at({});
    }

    void particles_at(const Qubits& sites);

    void single_particle();

    void all_particles();

    void checkerboard_particles();

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
