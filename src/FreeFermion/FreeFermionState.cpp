#include "FreeFermionState.h"
#include "Random.hpp"

bool FreeFermionState::projective_measurement(size_t i) {
  double c = occupation(i);
  bool outcome = randf() < c;
  forced_projective_measurement(i, outcome);
  return outcome;
}

double FreeFermionState::num_particles() const {
  double n = 0.0;
  for (size_t i = 0; i < L; i++) {
    n += occupation(i);
  }

  return n;
}

std::vector<double> FreeFermionState::occupation() const {
  std::vector<double> n(L);

  for (size_t i = 0; i < L; i++) {
    n[i] = occupation(i);
  }

  return n;
}
