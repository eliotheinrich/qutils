#include "PyQutils.hpp"

#include "Logger.hpp"
#include "CliffordState.h"
#include "FreeFermion.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

using namespace nanobind::literals;

template <typename T=double>
using ndarray = nanobind::ndarray<nanobind::numpy, T>;

template <typename T=double>
ndarray<T> to_ndarray(const std::vector<T>& values, const std::vector<size_t>& shape) {
  size_t k = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    k *= shape[i];
  }

  T* buffer = new T[k];
  std::move(values.begin(), values.end(), buffer); 

  nanobind::capsule owner(buffer, [](void* p) noexcept {
    delete static_cast<T*>(p);
  });

  return ndarray<T>(buffer, shape.size(), shape.data(), owner);
}

template <typename T>
static std::vector<size_t> get_shape(ndarray<T> arr) {
  size_t dim = arr.ndim();
  std::vector<size_t> shape(dim);
  for (size_t i = 0; i < dim; i++) {
    shape[i] = arr.shape(i);
  }

  return shape;
}

nanobind::bytes convert_bytes(const std::vector<char>& bytes) {
  nanobind::bytes nb_bytes(bytes.data(), bytes.size());
  return nb_bytes;
}

std::vector<char> convert_bytes(const nanobind::bytes& bytes) {
  std::vector<char> bytes_vec;
  bytes_vec.reserve(bytes.size() + 1);
  bytes_vec.insert(bytes_vec.end(), bytes.c_str(), bytes.c_str() + bytes.size());
  bytes_vec.push_back('\0');
  return bytes_vec;
}

NB_MODULE(qutils_bindings, m) {
  m.def("log_info", &Logger::log_info);
  m.def("log_warning", &Logger::log_warning);
  m.def("log_info", &Logger::log_error);
  m.def("read_log", &Logger::read_log);

  m.def("seed_rng", &Random::seed_rng);

  nanobind::class_<BitString>(m, "BitString")
    .def(nanobind::init<uint32_t>())
    .def("to_integer", &BitString::to_integer)
    .def_static("from_bits", [](uint32_t num_bits, uint32_t z) { return BitString::from_bits(num_bits, z); })
    .def_ro("bits", &BitString::bits)
    .def_ro("num_bits", &BitString::num_bits)
    .def("__str__", [](BitString& bs) { return fmt::format("{}", bs); })
    .def("hamming_weight", &BitString::hamming_weight)
    .def("substring", [](const BitString& self, const std::vector<uint32_t>& sites, bool keep_sites) { return self.substring(sites, keep_sites); }, "sites"_a, "keep"_a = true)
    .def("superstring", &BitString::superstring)
    .def("get", &BitString::get)
    .def("set", &BitString::set)
    .def("size", &BitString::size);

  nanobind::class_<PauliString>(m, "PauliString")
    .def(nanobind::init<const std::string&>())
    .def(nanobind::init<const PauliString&>())
    .def_ro("num_qubits", &PauliString::num_qubits)
    .def_static("from_bits", [](uint32_t num_qubits, uint32_t z) { return PauliString::from_bitstring(num_qubits, z); })
    .def_static("rand", [](uint32_t num_qubits) { return PauliString::rand(num_qubits); })
    .def_static("randh", [](uint32_t num_qubits) { return PauliString::randh(num_qubits); })
    .def("__str__", &PauliString::to_string_ops)
    .def("__mul__", &PauliString::operator*)
    .def("__rmul__", &PauliString::operator*)
    .def("__eq__", &PauliString::operator==)
    .def("__neq__", &PauliString::operator!=)
    .def("__getitem__", [](PauliString& self, size_t i) {
      if (i < self.num_qubits) {
        bool x = self.get_x(i);
        bool z = self.get_z(i);
        if (x && z) {
          return "Y";
        } else if (x && !z) {
          return "X";
        } else if (!x && z) {
          return "Z";
        } else {
          return "I";
        }
      } else {
        throw nanobind::index_error("Invalid index.");
      }
    })
    .def("to_matrix", [](PauliString& self) { return self.to_matrix(); })
    .def("to_projector", [](PauliString& self) { 
      Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(1u << self.num_qubits, 1u << self.num_qubits);
      Eigen::MatrixXcd M = self.to_matrix();
      Eigen::MatrixXcd P = (I + M)/2.0;
      return P;
    })
    .def("substring", [](const PauliString& self, const std::vector<uint32_t>& sites) { return self.substring(sites, true); })
    .def("substring_retain", [](const PauliString& self, const std::vector<uint32_t>& sites) { return self.substring(sites, false); })
    .def("superstring", &PauliString::superstring)
    .def("x", &PauliString::x)
    .def("y", &PauliString::y)
    .def("z", &PauliString::z)
    .def("s", &PauliString::s)
    .def("sd", &PauliString::sd)
    .def("h", &PauliString::h)
    .def("cx", &PauliString::cx)
    .def("cy", &PauliString::cy)
    .def("cz", &PauliString::cz)
    .def("evolve", [](PauliString& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("commutes", &PauliString::commutes)
    .def("set_x", &PauliString::set_x)
    .def("set_x", [](PauliString& self, size_t i, size_t v) { self.set_x(i, static_cast<bool>(v)); })
    .def("set_z", &PauliString::set_z)
    .def("set_z", [](PauliString& self, size_t i, size_t v) { self.set_z(i, static_cast<bool>(v)); })
    .def("get_x", &PauliString::get_x)
    .def("get_z", &PauliString::get_z)
    .def("reduce", [](const PauliString& self, bool z) { 
        QuantumCircuit qc(self.num_qubits);
        std::vector<uint32_t> qubits(self.num_qubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        self.reduce(z, std::make_pair(&qc, qubits));
        return qc;
      }, "z"_a = true);

  nanobind::class_<ADAMOptimizer>(m, "ADAMOptimizer")
    .def("__init__", [](ADAMOptimizer *adam, 
      double learning_rate, double beta1, double beta2, double epsilon,
      bool noisy_gradients, double gradient_noise
    ) {

      new (adam) ADAMOptimizer(learning_rate, beta1, beta2, epsilon, noisy_gradients, gradient_noise);
    }, "learning_rate"_a=0.001, "beta1"_a=0.9, "beta2"_a=0.999, "epsilon"_a=1e-8, "noisy_gradients"_a=false, "gradient_noise"_a=0.01)
  .def("reset", &ADAMOptimizer::reset)
  .def("step", &ADAMOptimizer::step);

  using MeasurementTuple = std::variant<
    std::tuple<Qubits, PauliString>, 
    std::tuple<Qubits, std::optional<double>, PauliString>
  >;
  nanobind::class_<QuantumCircuit>(m, "QuantumCircuit")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<uint32_t, uint32_t>())
    .def(nanobind::init<QuantumCircuit&>())
    .def("num_qubits", &QuantumCircuit::get_num_qubits)
    .def("__str__", &QuantumCircuit::to_string)
    .def("num_params", &QuantumCircuit::get_num_parameters)
    .def("bind_parameters", &QuantumCircuit::bind_parameters)
    .def("bind_measurement_outcomes", &QuantumCircuit::bind_measurement_outcomes)
    .def("length", &QuantumCircuit::length)
    .def("get_measurement_map", &QuantumCircuit::get_measurement_map)
    .def("get_parameter_map", &QuantumCircuit::get_parameter_map)
    .def("mzr", [](QuantumCircuit& self, uint32_t q, TargetOpt target) { 
      self.mzr(q, target);
    }, "qubit"_a, "target"_a=nanobind::none())
    .def("add_measurement", [](QuantumCircuit& self, const Qubits& qubits, const PauliString& pauli, std::optional<bool> outcome, TargetOpt target) {
      self.add_measurement(qubits, pauli, outcome, target);
    }, "qubits"_a, "pauli"_a, "outcome"_a=nanobind::none(), "target"_a=nanobind::none())
    .def("wmzr", [](QuantumCircuit& self, uint32_t q, std::optional<double> beta, TargetOpt target) { 
      self.wmzr(q, beta, target);
    }, "qubit"_a, "beta"_a=nanobind::none(), "target"_a=nanobind::none())
    .def("add_weak_measurement", [](QuantumCircuit& self, const Qubits& qubits, double beta, const PauliString& pauli, std::optional<bool> outcome) {
      self.add_weak_measurement(qubits, beta, pauli, outcome);
    }, "qubits"_a, "beta"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("add_gate", [](QuantumCircuit& self, const Eigen::MatrixXcd& gate, const Qubits& qubits, ControlOpt control) { 
      self.add_gate(gate, qubits, control); 
    }, "gate"_a, "qubits"_a, "control"_a=nanobind::none())
    .def("add_gate", [](QuantumCircuit& self, const Eigen::MatrixXcd& gate, uint32_t q, ControlOpt control) { 
      self.add_gate(gate, q, control); 
    }, "gate"_a, "qubit"_a, "control"_a=nanobind::none())
    .def("add_gate", [](QuantumCircuit& self, const FreeFermionGate& gate, ControlOpt control) {
      self.add_gate(gate, control);
    }, "gate"_a, "control"_a=nanobind::none())
    .def("add_gate", [](QuantumCircuit& self, const MajoranaGate& gate, ControlOpt control) {
      self.add_gate(gate, control);
    }, "gate"_a, "control"_a=nanobind::none())
    .def("add_gate", [](QuantumCircuit& self, const CommutingHamiltonianGate& gate, ControlOpt control) {
      self.add_gate(gate, control);
    }, "gate"_a, "control"_a=nanobind::none())
    .def("append", [](QuantumCircuit& self, const QuantumCircuit& other, const std::optional<Qubits>& qubits) { 
      if (qubits) {
        self.append(other, qubits.value());
      } else {
        self.append(other); 
      }
    }, "circuit"_a, "qubits"_a = nanobind::none())
    .def("h", &QuantumCircuit::h, "q"_a, "control"_a=nanobind::none())
    .def("x", &QuantumCircuit::x, "q"_a, "control"_a=nanobind::none())
    .def("y", &QuantumCircuit::y, "q"_a, "control"_a=nanobind::none())
    .def("z", &QuantumCircuit::z, "q"_a, "control"_a=nanobind::none())
    .def("s", &QuantumCircuit::s, "q"_a, "control"_a=nanobind::none())
    .def("sd", &QuantumCircuit::sd, "q"_a, "control"_a=nanobind::none())
    .def("t", &QuantumCircuit::t, "q"_a, "control"_a=nanobind::none())
    .def("td", &QuantumCircuit::td, "q"_a, "control"_a=nanobind::none())
    .def("sqrtX", &QuantumCircuit::sqrtX, "q"_a, "control"_a=nanobind::none())
    .def("sqrtY", &QuantumCircuit::sqrtY, "q"_a, "control"_a=nanobind::none())
    .def("sqrtZ", &QuantumCircuit::sqrtZ, "q"_a, "control"_a=nanobind::none())
    .def("sqrtXd", &QuantumCircuit::sqrtXd, "q"_a, "control"_a=nanobind::none())
    .def("sqrtYd", &QuantumCircuit::sqrtYd, "q"_a, "control"_a=nanobind::none())
    .def("sqrtZd", &QuantumCircuit::sqrtZd, "q"_a, "control"_a=nanobind::none())
    .def("cx", &QuantumCircuit::cx, "q1"_a, "q2"_a, "control"_a=nanobind::none())
    .def("cy", &QuantumCircuit::cy, "q1"_a, "q2"_a, "control"_a=nanobind::none())
    .def("cz", &QuantumCircuit::cz, "q1"_a, "q2"_a, "control"_a=nanobind::none())
    .def("swap", &QuantumCircuit::swap, "q1"_a, "q2"_a, "control"_a=nanobind::none())
    .def("rx", [](QuantumCircuit& self, uint32_t q, std::optional<double> theta_opt, ControlOpt control) { 
      self.rx(q, theta_opt, control); 
    }, "q"_a, "theta"_a=nanobind::none(), "control"_a=nanobind::none())
    .def("ry", [](QuantumCircuit& self, uint32_t q, std::optional<double> theta_opt, ControlOpt control) { 
      self.ry(q, theta_opt, control); 
    }, "q"_a, "theta"_a=nanobind::none(), "control"_a=nanobind::none())
    .def("rz", [](QuantumCircuit& self, uint32_t q, std::optional<double> theta_opt, ControlOpt control) { 
      self.rz(q, theta_opt, control); 
    }, "q"_a, "theta"_a=nanobind::none(), "control"_a=nanobind::none())
    .def("rp", [](QuantumCircuit& self, const Qubits& qubits, const PauliString& pauli, std::optional<double> theta_opt, ControlOpt control) {
      self.rp(qubits, pauli, theta_opt, control); 
    }, "qubits"_a, "pauli"_a, "theta"_a=nanobind::none(), "control"_a=nanobind::none())
    .def("random_clifford", [](QuantumCircuit& self, const std::vector<uint32_t>& qubits) {
      self.random_clifford(qubits);
    })
    .def("get_measurement", [](const QuantumCircuit& self, size_t i) -> MeasurementTuple {
      auto m = self.get_measurement(i);
      return std::visit(quantumcircuit_utils::overloaded{
        [](const Measurement& m) -> MeasurementTuple {
          return std::make_tuple(m.qubits, m.get_pauli());
        },
        [](const WeakMeasurement& m) -> MeasurementTuple {
          return std::make_tuple(m.qubits, m.beta, m.get_pauli());
        }
      }, m);
    })
    .def("cl_not", &QuantumCircuit::cl_not)
    .def("cl_and", &QuantumCircuit::cl_and)
    .def("cl_or", &QuantumCircuit::cl_or)
    .def("cl_xor", &QuantumCircuit::cl_xor)
    .def("cl_nand", &QuantumCircuit::cl_nand)
    .def("cl_clear", &QuantumCircuit::cl_clear)
    .def("is_clifford", &QuantumCircuit::is_clifford)
    .def("erase", &QuantumCircuit::erase)
    .def("insert", &QuantumCircuit::insert)
    .def("adjoint", [](QuantumCircuit& self, const std::optional<std::vector<double>> params) { return self.adjoint(params); }, "params"_a = nanobind::none())
    .def("reverse", &QuantumCircuit::reverse)
    .def("conjugate", &QuantumCircuit::conjugate)
    .def("to_matrix", [](QuantumCircuit& self, const std::optional<std::vector<double>> params) { return self.to_matrix(params); }, "params"_a = nanobind::none());

  m.def("random_clifford", [](uint32_t num_qubits) { 
    return random_clifford(num_qubits);
  });
  m.def("single_qubit_random_clifford", [](uint32_t r) {
    QuantumCircuit qc(1);
    single_qubit_clifford_impl(qc, 0, r % 24);
    return qc;
  });
  m.def("generate_haar_circuit", &generate_haar_circuit);
  m.def("hardware_efficient_ansatz", &hardware_efficient_ansatz);
  m.def("haar_unitary", [](uint32_t num_qubits) { return haar_unitary(num_qubits); });

  nanobind::class_<EntanglementEntropyState>(m, "EntanglementEntropyState")
    .def("entanglement", &EntanglementEntropyState::entanglement)
    .def("get_entanglement", &EntanglementEntropyState::get_entanglement<double>);

  nanobind::class_<MagicQuantumState, EntanglementEntropyState>(m, "QuantumState")
    .def("num_qubits", &MagicQuantumState::get_num_qubits)
    .def("__str__", &MagicQuantumState::to_string)
    .def("__getstate__", [](const MagicQuantumState& self) { return convert_bytes(self.serialize()); })
    .def("h", &MagicQuantumState::h)
    .def("x", &MagicQuantumState::x)
    .def("y", &MagicQuantumState::y)
    .def("z", &MagicQuantumState::z)
    .def("s", &MagicQuantumState::s)
    .def("sd", &MagicQuantumState::sd)
    .def("t", &MagicQuantumState::t)
    .def("td", &MagicQuantumState::td)
    .def("sqrtX", &MagicQuantumState::sqrtX)
    .def("sqrtY", &MagicQuantumState::sqrtY)
    .def("sqrtZ", &MagicQuantumState::sqrtZ)
    .def("sqrtXd", &MagicQuantumState::sqrtXd)
    .def("sqrtYd", &MagicQuantumState::sqrtYd)
    .def("sqrtZd", &MagicQuantumState::sqrtZd)
    .def("cx", &MagicQuantumState::cx)
    .def("cy", &MagicQuantumState::cy)
    .def("cz", &MagicQuantumState::cz)
    .def("swap", &MagicQuantumState::swap)
    .def("random_clifford", &MagicQuantumState::random_clifford)
    .def("partial_trace", [](MagicQuantumState& self, const Qubits& qubits) { return std::dynamic_pointer_cast<MagicQuantumState>(self.partial_trace(qubits)); })
    .def("expectation", [](const MagicQuantumState& self, const PauliString& pauli) { return self.expectation(pauli); })
    .def("expectation", [](const MagicQuantumState& self, const BitString& bits, std::optional<Qubits> support) { return self.expectation(bits, support); }, "bits"_a, "support"_a = nanobind::none())
    .def("expectation", [](const MagicQuantumState& self, const SparsePauliObs& obs) { return self.expectation(obs); })
    .def("probabilities", [](const MagicQuantumState& self) { 
      std::vector<size_t> shape = {self.basis};
      return to_ndarray(self.probabilities(), shape); 
    })
    .def("marginal_probabilities", [](const MagicQuantumState& self, const std::vector<Qubits>& qubits) {
      std::vector<QubitSupport> supports;
      for (const auto& q : qubits) {
        supports.push_back(q);
      }
      return self.marginal_probabilities(supports);
    })
    .def("partial_probabilities", [](const MagicQuantumState& self, const std::vector<Qubits>& qubits) {
      std::vector<QubitSupport> supports;
      for (const auto& q : qubits) {
        supports.push_back(q);
      }
      return self.partial_probabilities(supports);
    })
    .def("purity", &MagicQuantumState::purity)
    .def("mzr", [](MagicQuantumState& self, uint32_t q, std::optional<bool> outcome) {
      return self.measure(Measurement::computational_basis(q, outcome));
    }, "qubit"_a, "outcome"_a=std::nullopt)
    .def("measure", [](MagicQuantumState& self, const Qubits& qubits, const PauliString& pauli, std::optional<bool> outcome) {
      return self.measure(Measurement(qubits, pauli, outcome));
    }, "qubits"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("weak_measure", [](MagicQuantumState& self, const Qubits& qubits, double beta, const PauliString& pauli, std::optional<bool> outcome) {
      return self.weak_measure(WeakMeasurement(qubits, beta, pauli, outcome));
    }, "qubits"_a, "beta"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("entanglement", &MagicQuantumState::entanglement, "qubits"_a, "index"_a)
    .def("sample_bitstrings", &MagicQuantumState::sample_bitstrings)
    .def("sample_paulis", &MagicQuantumState::sample_paulis)
    .def("sample_paulis_exact", &MagicQuantumState::sample_paulis_exact)
    .def("sample_paulis_exhaustive", &MagicQuantumState::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", [](MagicQuantumState& self, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, PyMutationFunc py_mutation) {
      auto mutation = convert_from_pyfunc(py_mutation);
      return self.sample_paulis_montecarlo({}, num_samples, equilibration_timesteps, prob, mutation);
    })
    .def_static("calculate_magic_mutual_information_from_samples", [](MagicQuantumState& self, const MutualMagicAmplitudes& samples2, const MutualMagicAmplitudes& samples4) {
      return self.calculate_magic_mutual_information_from_samples(samples2, samples4);
    })
    .def("magic_mutual_information_samples_exact", &MagicQuantumState::magic_mutual_information_samples_exact)
    .def("magic_mutual_information_samples", &MagicQuantumState::magic_mutual_information_samples_montecarlo)
    .def("magic_mutual_information_exhaustive", &MagicQuantumState::magic_mutual_information_exhaustive)
    .def("bipartite_magic_mutual_information_samples_exact", &MagicQuantumState::bipartite_magic_mutual_information_samples_exact)
    .def("bipartite_magic_mutual_information_samples_montecarlo", &MagicQuantumState::bipartite_magic_mutual_information_samples_montecarlo)
    .def("bipartite_magic_mutual_information_exhaustive", &MagicQuantumState::bipartite_magic_mutual_information_exhaustive);

  m.def("renyi_entropy", &renyi_entropy);
  m.def("estimate_renyi_entropy", &estimate_renyi_entropy);
  m.def("estimate_mutual_renyi_entropy", &estimate_mutual_renyi_entropy);

  nanobind::class_<Statevector, MagicQuantumState>(m, "Statevector")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def(nanobind::init<Statevector>())
    .def(nanobind::init<MatrixProductState>())
    .def("__init__", [](Statevector *psi, const std::vector<std::complex<double>>& data) {
      Eigen::VectorXcd data_(data.size());
      for (size_t i = 0; i < data.size(); i++) {
        data_(i) = data[i];
      }

      new (psi) Statevector(data_);
    })
    .def("__setstate__", [](Statevector& self, const nanobind::bytes& bytes) { 
      new (&self) Statevector();
      self.deserialize(convert_bytes(bytes)); 
    })
    .def_ro("data", &Statevector::data)
    .def("normalize", &Statevector::normalize)
    .def("inner", &Statevector::inner)
    .def("expectation_m", [](Statevector& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc);
      }
    }, "circuit"_a, "params"_a=nanobind::none())
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, const Qubits& qubits, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, qubits, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc, qubits);
      }
    }, "circuit"_a, "qubits"_a, "params"_a=nanobind::none())
    .def("evolve", [](Statevector& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](Statevector& self, const PauliString& pauli, const Qubits& qubits) { self.QuantumState::evolve(pauli, qubits); })
    .def("evolve", [](Statevector& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<DensityMatrix, MagicQuantumState>(m, "DensityMatrix")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def("__setstate__", [](DensityMatrix& self, const nanobind::bytes& bytes) { 
      new (&self) DensityMatrix();
      self.deserialize(convert_bytes(bytes)); 
    })
    .def_ro("data", &DensityMatrix::data)
    .def("expectation_matrix", [](DensityMatrix& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc);
      }
    }, "circuit"_a, "params"_a=nanobind::none())
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, const Qubits& qubits, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, qubits, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc, qubits);
      }
    }, "circuit"_a, "qubits"_a, "params"_a=nanobind::none())
    .def("evolve", [](DensityMatrix& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](DensityMatrix& self, const PauliString& pauli, const Qubits& qubits) { self.QuantumState::evolve(pauli, qubits); })
    .def("evolve", [](DensityMatrix& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductState, MagicQuantumState>(m, "MatrixProductState")
    .def(nanobind::init<uint32_t, uint32_t, double>(), "num_qubits"_a, "max_bond_dimension"_a, "sv_threshold"_a=1e-9)
    .def(nanobind::init<const MatrixProductState&>())
    .def("__str__", &MatrixProductState::to_string)
    .def("__setstate__", [](MatrixProductState& self, const nanobind::bytes& bytes) { 
      new (&self) MatrixProductState(1, 2);
      self.deserialize(convert_bytes(bytes)); 
    })
    .def("print_mps", &MatrixProductState::print_mps)
    .def("set_debug_level", &MatrixProductState::set_debug_level)
    .def("set_orthogonality_level", &MatrixProductState::set_orthogonality_level)
    .def("bond_dimension_at_site", &MatrixProductState::bond_dimension)
    .def("singular_values", [](MatrixProductState& self, uint32_t q) { 
      std::vector<double> singular_values = self.singular_values(q);
      std::vector<size_t> shape = {singular_values.size()};
      return to_ndarray(singular_values, shape);
    }) 
    .def("tensor", [](MatrixProductState& self, uint32_t q) { 
      auto [shape, tensor] = self.tensor(q);
      return to_ndarray(tensor, shape); 
    })
    .def("get_logged_truncerr", [](MatrixProductState& self) { 
      std::vector<double> truncerr = self.get_logged_truncerr();
      std::vector<size_t> shape = {truncerr.size()};
      return to_ndarray(truncerr, shape); 
    })
    .def("trace", &MatrixProductState::trace)
    .def("expectation_matrix", [](MatrixProductState& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("concatenate", &MatrixProductState::concatenate)
    .def("conjugate", &MatrixProductState::conjugate)
    .def("inner", &MatrixProductState::inner)
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc);
      }
    }, "circuit"_a, "params"_a=nanobind::none())
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, const Qubits& qubits, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, qubits, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc, qubits);
      }
    }, "circuit"_a, "qubits"_a, "params"_a=nanobind::none())
    .def("evolve", [](MatrixProductState& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](MatrixProductState& self, const PauliString& pauli, const Qubits& qubits) { self.QuantumState::evolve(pauli, qubits); })
    .def("evolve", [](MatrixProductState& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  m.def("ising_ground_state", &MatrixProductState::ising_ground_state, "num_qubits"_a, "h"_a, "bond_dimension"_a=16, "sv_threshold"_a=1e-8, "num_sweeps"_a=10);
  m.def("xxz_ground_state", &MatrixProductState::xxz_ground_state, "num_qubits"_a, "delta"_a, "bond_dimension"_a=16, "sv_threshold"_a=1e-8, "num_sweeps"_a=10);
  m.def("spin_chain_ground_state", &MatrixProductState::spin_chain_ground_state, "num_qubits"_a, "Jx"_a, "Jy"_a, "Jz"_a, "hx"_a=nanobind::none(), "hy"_a=nanobind::none(), "hz"_a=nanobind::none(), "bond_dimension"_a=16, "sv_threshold"_a=1e-8, "num_sweeps"_a=10);

  nanobind::class_<QuantumCHPState, EntanglementEntropyState>(m, "QuantumCHPState")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<uint32_t, bool>())
    .def_ro("num_qubits", &QuantumCHPState::num_qubits)
    .def("set_print_mode", &QuantumCHPState::set_print_mode)
    .def("__str__", &QuantumCHPState::to_string)
    .def("__getstate__", [](const QuantumCHPState& self) { return convert_bytes(self.serialize()); })
    .def("__setstate__", [](QuantumCHPState& self, const nanobind::bytes& bytes) { 
      new (&self) QuantumCHPState();
      self.deserialize(convert_bytes(bytes)); })
    .def("tableau", [](QuantumCHPState& self) { return self.tableau->to_matrix(); })
    .def("evolve", [](QuantumCHPState& self, const QuantumCircuit& qc, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc);
      }
    }, "circuit"_a, "params"_a=nanobind::none())
    .def("evolve", [](QuantumCHPState& self, const QuantumCircuit& qc, const Qubits& qubits, std::optional<std::map<std::string, Parameter>> params) -> EvolveResult { 
      if (params) {
        return self.evolve(qc, qubits, EvolveOpts(params.value())); 
      } else {
        return self.evolve(qc, qubits);
      }
    }, "circuit"_a, "qubits"_a, "params"_a=nanobind::none())
    .def("evolve", [](QuantumCHPState& self, const PauliString& pauli, const Qubits& qubits) { self.QuantumState::evolve(pauli, qubits); })
    .def("stabilizers", [](const QuantumCHPState& self) { return self.stabilizers(); })
    .def("get_stabilizer", [](const QuantumCHPState& self, size_t i) { 
        if (i < self.num_qubits) {
          return self.get_stabilizer(i); 
        } else {
          throw nanobind::index_error("Index out of bounds.");
        }
    })
    .def("get_destabilizer", [](const QuantumCHPState& self, size_t i) { 
        if (i < self.num_qubits) {
          return self.get_destabilizer(i); 
        } else {
          throw nanobind::index_error("Index out of bounds.");
        }
    })
    .def("mzr_deterministic", [](const QuantumCHPState& self, size_t i) {
      auto [det, _] = self.tableau->mzr_deterministic(i);
      return det;
    })
    .def("rref", &QuantumCHPState::rref)
    .def("xrref", &QuantumCHPState::xrref)
    .def("rank", [](QuantumCHPState& self, const Qubits& qubits) { return self.partial_rank(qubits); })
    .def("xrank", [](QuantumCHPState& self, const Qubits& qubits) { return self.partial_xrank(qubits); })
    .def("h", [](QuantumCHPState& self, uint32_t q) { self.h(q); })
    .def("s", [](QuantumCHPState& self, uint32_t q) { self.s(q); })
    .def("sd", [](QuantumCHPState& self, uint32_t q) { self.sd(q); })
    .def("x", [](QuantumCHPState& self, uint32_t q) { self.x(q); })
    .def("y", [](QuantumCHPState& self, uint32_t q) { self.y(q); })
    .def("z", [](QuantumCHPState& self, uint32_t q) { self.z(q); })
    .def("sqrtx", [](QuantumCHPState& self, uint32_t q) { self.sqrtx(q); })
    .def("sqrty", [](QuantumCHPState& self, uint32_t q) { self.sqrty(q); })
    .def("sqrtz", [](QuantumCHPState& self, uint32_t q) { self.sqrtz(q); })
    .def("sqrtxd", [](QuantumCHPState& self, uint32_t q) { self.sqrtxd(q); })
    .def("sqrtyd", [](QuantumCHPState& self, uint32_t q) { self.sqrtyd(q); })
    .def("sqrtzd", [](QuantumCHPState& self, uint32_t q) { self.sqrtzd(q); })
    .def("cx", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.cx(q1, q2); })
    .def("cy", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.cy(q1, q2); })
    .def("cz", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.cz(q1, q2); })
    .def("swap", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.swap(q1, q2); })
    .def("measure", [](QuantumCHPState& self, const Qubits& qubits, const PauliString& pauli, std::optional<bool> outcome) { 
      return self.expectation(pauli); 
    }, "qubits"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("expectation", [](QuantumCHPState& self, const PauliString& pauli) { return self.expectation(pauli); })
    .def("mxr", [](QuantumCHPState& self, uint32_t q) { return self.mxr(q); })
    .def("myr", [](QuantumCHPState& self, uint32_t q) { return self.myr(q); })
    .def("mzr", [](QuantumCHPState& self, uint32_t q) { return self.mzr(q); })
    .def("mxr_expectation", [](QuantumCHPState& self, uint32_t q) { return self.mxr_expectation(q); })
    .def("myr_expectation", [](QuantumCHPState& self, uint32_t q) { return self.myr_expectation(q); })
    .def("mzr_expectation", [](QuantumCHPState& self, uint32_t q) { return self.mzr_expectation(q); })
    .def("to_statevector", &QuantumCHPState::to_statevector)
    .def("entanglement", &QuantumCHPState::entanglement, "qubits"_a, "index"_a=2)
    .def("get_entanglement", [](QuantumCHPState& self) {
      return self.get_entanglement<int>(2u);
    })
    .def("random_clifford", &QuantumCHPState::random_clifford);

  nanobind::class_<CliffordTable<QuantumCircuit>>(m, "CliffordTable")
    .def("__init__", [](CliffordTable<QuantumCircuit> *t, const std::vector<PauliString>& p1, const std::vector<PauliString>& p2) {
        if (p1.size() != p2.size()) {
          throw std::runtime_error("Mismatched length of PauliStrings in filter function for CliffordTable.");
        }

        auto filter = [&p1, &p2](const QuantumCircuit& qc) -> bool {
          for (size_t i = 0; i < p1.size(); i++) {
            PauliString q1 = p1[i];
            PauliString q2 = p2[i];
            qc.apply(q1);
            if (q1 != q2) {
              return false;
            }
          }

          return true;
        };
        new (t) CliffordTable<QuantumCircuit>(filter); 
    })
    .def("circuits", &CliffordTable<QuantumCircuit>::get_circuits)
    .def("num_elements", &CliffordTable<QuantumCircuit>::num_elements)
    .def("random_circuit", [](CliffordTable<QuantumCircuit>& self) {
      QuantumCircuit qc(2);
      self.apply_random({0, 1}, qc);
      return qc;
    });

  nanobind::class_<FreeFermionGate>(m, "FreeFermionGate")
    .def("__init__", [](FreeFermionGate* gate, 
      size_t num_qubits, std::optional<double> t
    ) {
      new (gate) FreeFermionGate(num_qubits, t);
    }, "num_qubits"_a, "t"_a=nanobind::none())
    .def(nanobind::init<const FreeFermionGate&>())
    .def(nanobind::init<const MajoranaGate&>())
    .def("set_t", &FreeFermionGate::set_t)
    .def("add_term", &FreeFermionGate::add_term, "i"_a, "j"_a, "a"_a, "adj"_a=true)
    .def("to_gate", [](const FreeFermionGate& self) { return self.to_gate()->define(); })
    .def("__add__", &FreeFermionGate::combine)
    .def("__str__", &FreeFermionGate::to_string);

  nanobind::class_<MajoranaGate>(m, "MajoranaGate")
    .def("__init__", [](MajoranaGate* gate, 
      size_t num_qubits, std::optional<double> t
    ) {
      new (gate) MajoranaGate(num_qubits, t);
    }, "num_qubits"_a, "t"_a=nanobind::none())
    .def(nanobind::init<const MajoranaGate&>())
    .def("set_t", &MajoranaGate::set_t)
    .def("add_term", &MajoranaGate::add_term, "i"_a, "j"_a, "a"_a)
    .def("to_gate", [](const MajoranaGate& self) { return self.to_gate()->define(); })
    .def("__add__", &MajoranaGate::combine)
    .def("__str__", &MajoranaGate::to_string);

  nanobind::class_<CommutingHamiltonianGate>(m, "CommutingHamiltonianGate")
    .def("__init__", [](CommutingHamiltonianGate* gate, 
      size_t num_qubits, std::optional<double> t
    ) {
      new (gate) CommutingHamiltonianGate(num_qubits, t);
    }, "num_qubits"_a, "t"_a=nanobind::none())
    .def(nanobind::init<const CommutingHamiltonianGate&>())
    .def("add_term", &CommutingHamiltonianGate::add_term, "a"_a, "pauli"_a, "qubits"_a)
    .def("__str__", &CommutingHamiltonianGate::to_string);

  nanobind::class_<GaussianState, MagicQuantumState>(m, "GaussianState")
    .def(nanobind::init<uint32_t>())
    .def("__init__", [](GaussianState* state, 
      size_t num_qubits, std::optional<Qubits> sites
    ) {
      new (state) GaussianState(num_qubits, sites);
    }, "num_qubits"_a, "sites"_a=nanobind::none())
    .def("__setstate__", [](GaussianState& self, const nanobind::bytes& bytes) { 
      new (&self) GaussianState();
      self.deserialize(convert_bytes(bytes)); 
    })
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, std::optional<std::map<std::string, Parameter>> params) { 
      if (params) {
        self.evolve(qc, EvolveOpts(params.value())); 
      } else {
        self.evolve(qc);
      }
    }, "circuit"_a, "params"_a=nanobind::none())
    .def("evolve", [](MagicQuantumState& self, const QuantumCircuit& qc, const Qubits& qubits, std::optional<std::map<std::string, Parameter>> params) { 
      if (params) {
        self.evolve(qc, qubits, EvolveOpts(params.value())); 
      } else {
        self.evolve(qc, qubits);
      }
    }, "circuit"_a, "qubits"_a, "params"_a=nanobind::none())
    .def("evolve", [](GaussianState& state, const FreeFermionGate& gate) {
      state.evolve(gate);
    })
    .def("evolve", [](GaussianState& state, const MajoranaGate& gate) {
      state.evolve(FreeFermionGate(gate));
    })
    .def("num_particles", &GaussianState::num_particles)
    .def("covariance_matrix", &GaussianState::covariance_matrix)
    .def("majorana_covariance_matrix", &GaussianState::majorana_covariance_matrix)
    .def("occupation", [](GaussianState& self, size_t i) { return self.occupation(i); });
}
