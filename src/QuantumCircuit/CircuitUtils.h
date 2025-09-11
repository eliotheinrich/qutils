#pragma once

#include <map>
#include <optional>
#include <utility>
#include <random>

#include <Eigen/Dense>

#include "Support.hpp"

namespace quantumcircuit_utils {
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

template <typename T>
std::map<T, size_t> reverse_map(const std::vector<T>& vec) {
  std::map<T, size_t> reversed_map;
  for (size_t i = 0; i < vec.size(); i++) {
    reversed_map[vec[i]] = i;
  }

  return reversed_map;
}

bool qargs_unique(const Qubits& qubits);
Qubits parse_qargs_opt(const std::optional<Qubits>& qubits_opt, uint32_t num_qubits);
std::pair<uint32_t, uint32_t> get_targets(uint32_t d, uint32_t q, uint32_t num_qubits);
Qubits complement(const Qubits& qubits, size_t num_qubits);

Eigen::MatrixXcd haar_unitary(uint32_t num_qubits);

Eigen::MatrixXcd random_real_unitary();

Eigen::MatrixXcd embed_unitary(const Eigen::MatrixXcd &gate, const Qubits &qubits, uint32_t total_qubits);

Eigen::MatrixXcd normalize_unitary(Eigen::MatrixXcd &unitary);

class ADAMOptimizer {
	private:
		double learning_rate;
		double beta1;
		double beta2;
		double epsilon;
		std::vector<double> m;
		std::vector<double> v;

		std::mt19937 generator;
		std::normal_distribution<double> noise_distribution;

		bool noisy_gradients;
		double gradient_noise;

    bool params_initialized;
    size_t num_params;
    
    void initialize(size_t num_params);

	public:
		int t;

		ADAMOptimizer(
			double learning_rate=0.001, double beta1=0.9, double beta2=0.99, double epsilon=1e-8,
			bool noisy_gradients=false, double gradient_noise=0.01
		);

    void reset();
    std::vector<double> step(const std::vector<double>& params, const std::vector<double>& gradients);
};
