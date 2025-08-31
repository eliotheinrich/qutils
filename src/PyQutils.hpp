#pragma once

#include "QuantumCircuit.h"
#include "QuantumState.h"

using PyMutationFunc = std::function<PauliString(PauliString)>;
inline PauliMutationFunc convert_from_pyfunc(PyMutationFunc func) {
  return [func](PauliString& p) { p = func(p); };
}
