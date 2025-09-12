#include "TableauSIMD.h"

static inline constexpr size_t binary_word_size() {
  return 8u*sizeof(binary_word);
}

inline binary_word _get(const binary_word* data, size_t j) {
  binary_word word = data[j / binary_word_size()];
  size_t bit_ind = j % binary_word_size();

  return (word >> bit_ind) & static_cast<binary_word>(1);
}

std::string binary_word_to_string(const binary_word* word, size_t length) {
  std::string s = "";
  for (size_t i = 0; i < length; i++) {
    s += fmt::format("{}", _get(word, i));
  }
  return s;
}

std::string binary_word_to_string_ops(const binary_word* word, size_t length) {
  std::string s = "";
  for (size_t i = 0; i < length/2; i++) {
    bool x = _get(word, 2*i);
    bool z = _get(word, 2*i+1);

    if (x && z) {
      s += "Y";
    } else if (x) {
      s += "X";
    } else if (z) {
      s += "Z";
    } else {
      s += "I";
    }
  }
  return s;
}


inline void _set(binary_word* data, size_t j, binary_word v) {
  size_t word_ind = j / binary_word_size();
  size_t bit_ind = j % binary_word_size();

  data[word_ind] = (data[word_ind] & ~(static_cast<binary_word>(1) << bit_ind)) | (v << bit_ind);
}

bool TableauSIMD::get(size_t i, size_t j) const {
  return _get(rows[i], j);
}

void TableauSIMD::set(size_t i, size_t j, binary_word v) {
  _set(rows[i], j, v);
}

inline size_t get_width(size_t num_bits) {
  return num_bits / binary_word_size() + static_cast<bool>(num_bits % binary_word_size());
}

TableauSIMD::TableauSIMD(uint32_t num_qubits) : TableauBase(num_qubits) {
  width = get_width(2*num_qubits);
  rows.resize(2*num_qubits + 1);
  for (auto& row : rows) {
    row = new binary_word[width];
  }

  for (uint32_t i = 0; i < num_qubits; i++) {
    reset(i);
    reset(i + num_qubits);
    set(i, 2*i, true);
    set(i + num_qubits, 2*i+1, true);
  }

  pwidth = get_width(2*num_qubits + 1);
  phase = new binary_word[pwidth];
}

Pauli TableauSIMD::get_pauli(size_t i, size_t j) const {
  bool x = _get(rows[i + num_qubits], 2*j);
  bool z = _get(rows[i + num_qubits], 2*j+1);
  if (x && z) {
    return Pauli::Y;
  } else if (x) {
    return Pauli::X;
  } else if (z) {
    return Pauli::Z;
  } else {
    return Pauli::I;
  } 
}


uint8_t TableauSIMD::get_phase(size_t i) const {
  return 2*_get(phase, i + num_qubits);
}

void TableauSIMD::reset(int i) {
  std::fill(rows[i], rows[i] + width, 0u);
}

void TableauSIMD::rowsum(int i, int j) {
  uint8_t s = 2*_get(phase, i) + 2*_get(phase, j);
  for (uint32_t k = 0; k < num_qubits; k++) {
    s += multiplication_phase(get_xz(j, k), get_xz(i, k));
  }
  
  for (size_t k = 0; k < width; k++) {
    rows[i][k] = rows[i][k] ^ rows[j][k];
  }

  _set(phase, i, s % 4 == 2);
}

TableauSIMD::~TableauSIMD() {
  for (auto ptr : rows) {
    delete[] ptr;
  }

  delete[] phase;
}

bool TableauSIMD::operator==(TableauSIMD& other) {
  if (num_qubits != other.num_qubits) {
    return false;
  }

  rref();
  other.rref();

  for (uint32_t i = 0; i < num_qubits; i++) {
    if (_get(phase, i) != _get(other.phase, i)) {
      return false;
    }

    for (uint32_t j = 0; j < 2*num_qubits; j++) {
      if (get(i + num_qubits, j) != other.get(i + num_qubits, j)) {
        return false;
      }
    }
  }

  return true;
}

void TableauSIMD::rref(const Qubits& sites) {
  uint32_t pivot_row = num_qubits;
  uint32_t row = num_qubits;

  for (uint32_t k = 0; k < 2*sites.size(); k++) {
    uint32_t c = sites[k % sites.size()];
    bool z = k < sites.size();
    bool found_pivot = false;
    for (uint32_t i = row; i < 2*num_qubits; i++) {
      if ((z && get(i, 2*c+1)) || (!z && get(i, 2*c))) {
        pivot_row = i;
        found_pivot = true;
        break;
      }
    }

    if (found_pivot) {
      std::swap(rows[row], rows[pivot_row]);

      for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
        if (i == row) {
          continue;
        }

        if ((z && get(i, 2*c+1)) || (!z && get(i, 2*c))) {
          rowsum(i, row);
        }
      }

      row += 1;
    } else {
      continue;
    }
  }
}

void TableauSIMD::xrref(const Qubits& sites) {
  uint32_t pivot_row = num_qubits;
  uint32_t row = num_qubits;

  for (uint32_t k = 0; k < 2*sites.size(); k++) {
    uint32_t c = sites[k % sites.size()];
    bool z = k < sites.size();
    bool found_pivot = false;
    for (uint32_t i = row; i < 2*num_qubits; i++) {
      if (!z && get(i, 2*c)) {
        pivot_row = i;
        found_pivot = true;
        break;
      }
    }

    if (found_pivot) {
      std::swap(rows[row], rows[pivot_row]);

      for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
        if (i == row) {
          continue;
        }

        if (!z && get(i, 2*c)) {
          rowsum(i, row);
        }
      }

      row += 1;
    } else {
      continue;
    }
  }
}

uint32_t TableauSIMD::xrank(const Qubits& sites) {
  xrref(sites);

  uint32_t r = 0;
  for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
    for (uint32_t j = 0; j < sites.size(); j++) {
      if (get(i, 2*sites[j])) {
        r++;
        break;
      }
    }
  }

  return r;
}

uint32_t TableauSIMD::rank(const Qubits& sites) {
  rref(sites);

  uint32_t r = 0;
  for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
    for (uint32_t j = 0; j < sites.size(); j++) {
      if (get(i, 2*sites[j]) || get(i, 2*sites[j]+1)) {
        r++;
        break;
      }
    }
  }

  return r;
}

void TableauSIMD::rref() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  rref(qubits);
}

void TableauSIMD::xrref() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  xrref(qubits);
}

uint32_t TableauSIMD::rank() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return rank(qubits);
}

uint32_t TableauSIMD::xrank() {
  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  return xrank(qubits);
}

//TableauSIMD TableauSIMD::partial_trace(const Qubits& qubits) {
//  rref(qubits);
//
//  TableauSIMD tableau_new;
//  tableau_new.num_qubits = num_qubits - qubits.size();
//
//  Qubits qubits_complement = to_qubits(support_complement(qubits, num_qubits));
//
//  for (const PauliString& stab : stabilizers) {
//    bool is_id = true;
//    for (size_t i = 0; i < qubits.size(); i++) {
//      if (stab.get_x(qubits[i]) || stab.get_z(qubits[i])) {
//        is_id = false;
//        break;
//      }
//    }
//
//    if (is_id) {
//      tableau_new.stabilizers.push_back(stab.substring(qubits_complement, true));
//    }
//  }
//
//  return tableau_new;
//}

double TableauSIMD::bitstring_amplitude(const BitString& bits) {
  if (bits.num_bits != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot evaluate a bitstring of {} bits on a TableauSIMD of {} qubits.", bits.num_bits, num_qubits));
  }

  // TODO replace with rank defficiency?
  double p = 1/std::pow(2.0, xrank() + num_qubits - num_qubits);

  bool in_support = true;
  for (size_t r = num_qubits; r < 2*num_qubits; r++) {
    // Need to check that every z-only stabilizer g acts on |z> as g|z> = |z>. 
    bool has_x = false;
    for (size_t i = 0; i < num_qubits; i++) {
      if (get(r, 2*i)) {
        has_x = true;
        break;
      }
    }

    if (has_x) {
      continue;
    }

    bool positive = true;
    for (size_t i = 0; i < num_qubits; i++) {
      if (get(r, 2*i+1) && bits.get(i)) {
        positive = !positive;
      }
    }

    if (positive != (_get(phase, r) == 0)) {
      in_support = false;
      break;
    }
  }

  return in_support ? p : 0.0;
}

std::string TableauSIMD::to_string(bool print_destabilizers) const {
  std::string s = "";
  if (print_destabilizers) {
    for (size_t i = 0; i < num_qubits; i++) {
      s += (i == 0) ? "[" : " ";
      s += (_get(phase, i) ? "-" : "+") + binary_word_to_string(rows[i], 2*num_qubits);
      s += (i == num_qubits - 1) ? "]" : "\n";
    }
    s += "\n";
  }

  for (size_t i = num_qubits; i < 2*num_qubits; i++) {
    s += (i == 0) ? "[" : " ";
    s += (_get(phase, i) ? "-" : "+") + binary_word_to_string(rows[i], 2*num_qubits);
    s += (i == num_qubits - 1) ? "]" : "\n";
  }

  return s;
}

std::string TableauSIMD::to_string_ops(bool print_destabilizers) const {
  std::string s = "";
  if (print_destabilizers) {
    for (size_t i = 0; i < num_qubits; i++) {
      s += (i == 0) ? "[" : " ";
      s += (_get(phase, i) ? "-" : "+") + binary_word_to_string_ops(rows[i], 2*num_qubits);
      s += (i == num_qubits - 1) ? "]" : "\n";
    }
    s += "\n";
  }

  for (size_t i = num_qubits; i < 2*num_qubits; i++) {
    s += (i == num_qubits) ? "[" : " ";
    s += (_get(phase, i) ? "-" : "+") + binary_word_to_string_ops(rows[i], 2*num_qubits);
    s += (i == 2*num_qubits - 1) ? "]" : "\n";
  }
  return s;
}

void TableauSIMD::h(uint32_t a) {
  validate_qubit(a);
  for (size_t i = 0; i < 2*num_qubits; i++) {
    uint8_t xza = get_xz(i, a);
    bool xa = (xza >> 0u) & 1u;
    bool za = (xza >> 1u) & 1u;

    constexpr bool h_phase_lookup[] = {0, 0, 0, 1};

    _set(phase, i, _get(phase, i) != h_phase_lookup[xza]);
    set(i, 2*a, za);
    set(i, 2*a+1, xa);
  }
}

void TableauSIMD::s(uint32_t a) {
  validate_qubit(a);
  for (size_t i = 0; i < 2*num_qubits; i++) {
    uint8_t xza = get_xz(i, a);
    bool xa = (xza >> 0u) & 1u;
    bool za = (xza >> 1u) & 1u;

    constexpr bool s_phase_lookup[] = {0, 0, 0, 1};
    _set(phase, i, (_get(phase, i) != s_phase_lookup[xza]));
    set(i, 2*a+1, xa != za);
  }
}

void TableauSIMD::cx(uint32_t a, uint32_t b) {
  validate_qubit(a);
  validate_qubit(b);
  for (size_t i = 0; i < 2*num_qubits; i++) {
    uint8_t xza = get_xz(i, a);
    bool xa = (xza >> 0u) & 1u;
    bool za = (xza >> 1u) & 1u;

    uint8_t xzb = get_xz(i, b);
    bool xb = (xzb >> 0u) & 1u;
    bool zb = (xzb >> 1u) & 1u;

    uint8_t bitcode = xzb + (xza << 2);

    constexpr bool cx_phase_lookup[] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    _set(phase, i, _get(phase, i) != cx_phase_lookup[bitcode]);
    set(i, 2*b, xa != xb);
    set(i, 2*a+1, za != zb);
  }
}

std::pair<bool, uint32_t> TableauSIMD::mzr_deterministic(uint32_t a) const {
  for (uint32_t p = num_qubits; p < 2*num_qubits; p++) {
    // Suitable p identified; outcome is random
    if (get(p, 2*a)) { 
      return std::pair(false, p);
    }
  }

  // No p found; outcome is deterministic
  return std::pair(true, 0);
}

MeasurementData TableauSIMD::mzr(uint32_t a, std::optional<bool> outcome) {
  validate_qubit(a);

  auto [deterministic, p] = mzr_deterministic(a);

  if (!deterministic) {
    bool b = outcome ? outcome.value() : randi() % 2;

    for (uint32_t i = 0; i < 2*num_qubits; i++) {
      if (i != p && get(i, 2*a)) {
        rowsum(i, p);
      }
    }

    std::swap(rows[p], rows[p - num_qubits]);

    reset(p);
    _set(phase, p, b);
    set(p, 2*a+1, true);

    return {b, 0.5};
  } else { // deterministic
    reset(2*num_qubits);
    for (uint32_t i = 0; i < num_qubits; i++) {
      if (get(i, 2*a)) {
        rowsum(2*num_qubits, i + num_qubits);
      }
    }

    bool b = _get(phase, 2*num_qubits);

    if (outcome) {
      if (b != outcome.value()) {
        throw std::runtime_error("Invalid forced measurement of QuantumCHPState.");
      }
    }
    
    return {b, 1.0};
  }
}
