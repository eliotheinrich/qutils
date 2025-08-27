# qutils
**qutils** is a library for simulation of quantum circuits, with a focus on single-shot efficiency and ease of use. Currently, qutils supports circuit evolution on statevectors, density matrices, matrix product states (https://arxiv.org/abs/quant-ph/0301063), stabilizer states (https://www.arxiv.org/abs/quant-ph/0406196), and free fermion dynamics. qutils is natively written in C++, but is primarily meant to be interfaced with a set of Python bindings.

# Installation
Right now, qutils should be installed locally into a conda environment. There are dependencies on OpenBLAS or MKL and the C++ version of ITensor (https://github.com/ITensor/ITensor). ITensor must be built and an environment variable ITENSOR_DIR should be created to point to the location of the installed itensor version at compile time. To build qutils, simply run
```
$ pip install .
```
from the root directory of the project to compile.

# Running
Executing quantum circuits in qutils is simple. For example,
```
$ first_circuit.py
import numpy as np

from qutils import QuantumCircuit, MatrixProductState, QuantumCHPState, PauliString

num_qubits = 32

qc = QuantumCircuit(num_qubits)
for q in range(num_qubits):
        qc.h(q)

for q in range(num_qubits - 1):
        qc.cx(q, q+1)

for q in range(num_qubits - 1):
    # Projective measurement in XX basis on qubits q, q+1
        qc.add_measurement([q, q+1], PauliString("XX"))

bond_dimension = 16
mps = MatrixProductState(num_qubits, bond_dimension)
mps.evolve(qc)

chp = QuantumCHPState(num_qubits)
chp.evolve(qc)

# P = XXII...
p = PauliString("I"*num_qubits)
p.set_x(0, 1)
p.set_x(1, 1)

# CHP state does not track phase, so only check absolute values
e1 = np.abs(mps.expectation(p))
e2 = np.abs(chp.expectation(p))
print(e1, e2)
```

For a full list of the available classes and methods, check out src/PyQutils.cpp.
