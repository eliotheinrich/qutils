#include "TableauSIMD.h"

#ifdef __AVX__
#include <immintrin.h>

inline binary_word _get(const binary_word* data, size_t j) {
  binary_word word = data[j / binary_word_size()];
  size_t bit_ind = j % binary_word_size();

  return (word >> bit_ind) & static_cast<binary_word>(1);
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

  pwidth = get_width(2*num_qubits + 1);
  phase = new binary_word[pwidth];

  for (uint32_t i = 0; i < num_qubits; i++) {
    reset(i);
    reset(i + num_qubits);
    set(i, 2*i, true);
    set(i + num_qubits, 2*i+1, true);
  }
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

PauliString TableauSIMD::get_stabilizer(size_t i) const {
  std::vector<Pauli> paulis(num_qubits);
  for (size_t j = 0; j < num_qubits; j++) {
    bool x = _get(rows[i + num_qubits], 2*j);
    bool z = _get(rows[i + num_qubits], 2*j+1);
    if (x && z) {
      paulis[j] = Pauli::Y;
    } else if (x) {
      paulis[j] = Pauli::X;
    } else if (z) {
      paulis[j] = Pauli::Z;
    } else {
      paulis[j] = Pauli::I;
    } 
  }

  return PauliString(paulis, 2*_get(phase, i + num_qubits));
}

PauliString TableauSIMD::get_destabilizer(size_t i) const {
  std::vector<Pauli> paulis(num_qubits);
  for (size_t j = 0; j < num_qubits; j++) {
    bool x = _get(rows[i], 2*j);
    bool z = _get(rows[i], 2*j+1);
    if (x && z) {
      paulis[j] = Pauli::Y;
    } else if (x) {
      paulis[j] = Pauli::X;
    } else if (z) {
      paulis[j] = Pauli::Z;
    } else {
      paulis[j] = Pauli::I;
    } 
  }

  return PauliString(paulis, 2*_get(phase, i));
}

uint8_t TableauSIMD::get_phase(size_t i) const {
  return 2*_get(phase, i + num_qubits);
}

void TableauSIMD::reset(int i) {
  std::fill(rows[i], rows[i] + width, 0u);
  _set(phase, i, false);
}

constexpr __m256i generate_phase_vector() {
  constexpr std::array<int, 16> table = generate_phase_table();
  alignas(32) int8_t bytes[32];

  for (int i = 0; i < 16; ++i) {
    bytes[i]    = table[i];
    bytes[i+16] = table[i];
  }
  return std::bit_cast<__m256i>(bytes);
}

constexpr size_t LANES = 256/binary_word_size();

constexpr bool WORD_64_BITS = (sizeof(binary_word) == 8);
constexpr bool WORD_32_BITS = (sizeof(binary_word) == 4);
constexpr bool WORD_16_BITS = (sizeof(binary_word) == 2);

inline __m256i slli(__m256i v, size_t i) {
  if constexpr (WORD_64_BITS) {
    return _mm256_slli_epi64(v, i);
  } else if constexpr (WORD_32_BITS) {
    return _mm256_slli_epi32(v, i);
  } else if constexpr (WORD_16_BITS) {
    return _mm256_slli_epi16(v, i);
  }
}

inline __m256i srli(__m256i v, size_t i) {
  if constexpr (WORD_64_BITS) {
    return _mm256_srli_epi64(v, i);
  } else if constexpr (WORD_32_BITS) {
    return _mm256_srli_epi32(v, i);
  } else if constexpr (WORD_16_BITS) {
    return _mm256_srli_epi16(v, i);
  }
}

inline __m256i set1(size_t i) {
  if constexpr (WORD_64_BITS) {
    return _mm256_set1_epi64x(i);
  } else if constexpr (WORD_32_BITS) {
    return _mm256_set1_epi32(i);
  } else if constexpr (WORD_16_BITS) {
    return _mm256_set1_epi16(i);
  }
}

void TableauSIMD::rowsum(int i, int j) {
  uint8_t s = 2*_get(phase, i) + 2*_get(phase, j);

  size_t p = width - width % LANES;
  constexpr __m256i phase_vector_table = generate_phase_vector();
  const __m256i mask = _mm256_set1_epi8(static_cast<char>(0b11));
  for (size_t k = 0; k < p; k += LANES) {
    __m256i xz1_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[i] + k));
    __m256i xz2_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[j] + k));

    __m256i phases = _mm256_setzero_si256();
    for (size_t s = 0; s < 4; s++) {
      __m256i index = _mm256_or_si256(_mm256_and_si256(xz1_vec, mask), slli(_mm256_and_si256(xz2_vec, mask), 2));
      phases = _mm256_add_epi8(phases, _mm256_shuffle_epi8(phase_vector_table, index));
      xz1_vec = srli(xz1_vec, 2);
      xz2_vec = srli(xz2_vec, 2);
    }

    alignas(32) int8_t temp_bytes[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp_bytes), phases);
    for (int l = 0; l < 32; ++l) {
      s += temp_bytes[l];
    }

    __m256i a = _mm256_loadu_si256((const __m256i*)(rows[i] + k));
    __m256i b = _mm256_loadu_si256((const __m256i*)(rows[j] + k));
    __m256i r = _mm256_xor_si256(a, b);
    _mm256_storeu_si256((__m256i*)(rows[i] + k), r);
  }

  // Tail loops
  for (uint32_t n = p*binary_word_size()/2; n < num_qubits; n++) {
    s += multiplication_phase(get_xz(j, n), get_xz(i, n));
  }

  for (uint32_t n = p; n < width; n++) {
    rows[i][n] ^= rows[j][n];
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
      swap(row, pivot_row);
      swap(row - num_qubits, pivot_row - num_qubits);

      for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
        if (i == row) {
          continue;
        }

        if ((z && get(i, 2*c+1)) || (!z && get(i, 2*c))) {
          rowsum(i, row);
          rowsum(row - num_qubits, i - num_qubits);
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
      swap(row, pivot_row);
      swap(row - num_qubits, pivot_row - num_qubits);

      for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
        if (i == row) {
          continue;
        }

        if (!z && get(i, 2*c)) {
          rowsum(i, row);
          rowsum(row - num_qubits, i - num_qubits);
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

std::string binary_word_to_string(const binary_word* word, size_t length) {
  std::string s = "";
  for (size_t i = 0; i < length; i++) {
    s += fmt::format("{}", _get(word, i));
  }
  return s;
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

inline __m256i load_words(size_t i, size_t word_ind, const std::vector<binary_word*>& rows) {
  static_assert(std::is_unsigned_v<binary_word>, "binary_word must be an unsigned integral type.");

  if constexpr (WORD_64_BITS) {
    return _mm256_setr_epi64x(rows[i+0][word_ind], rows[i+1][word_ind], rows[i+2][word_ind], rows[i+3][word_ind]);
  } else if constexpr (WORD_32_BITS) {
    return _mm256_setr_epi32(rows[i+0][word_ind], rows[i+1][word_ind], rows[i+2][word_ind], rows[i+3][word_ind],
                             rows[i+4][word_ind], rows[i+5][word_ind], rows[i+6][word_ind], rows[i+7][word_ind]);
  } else if constexpr (WORD_16_BITS) {
    return _mm256_setr_epi16(rows[i+0 ][word_ind], rows[i+1 ][word_ind], rows[i+2 ][word_ind], rows[i+3 ][word_ind],
                             rows[i+4 ][word_ind], rows[i+5 ][word_ind], rows[i+6 ][word_ind], rows[i+7 ][word_ind],
                             rows[i+8 ][word_ind], rows[i+9 ][word_ind], rows[i+10][word_ind], rows[i+11][word_ind],
                             rows[i+12][word_ind], rows[i+13][word_ind], rows[i+14][word_ind], rows[i+15][word_ind]);
  }
}

inline void save_words(size_t i, size_t word_ind, std::vector<binary_word*>& rows, __m256i words) {
  if constexpr (WORD_64_BITS) {
    rows[i+0][word_ind] = _mm256_extract_epi64(words, 0);
    rows[i+1][word_ind] = _mm256_extract_epi64(words, 1);
    rows[i+2][word_ind] = _mm256_extract_epi64(words, 2);
    rows[i+3][word_ind] = _mm256_extract_epi64(words, 3);
  } else if constexpr (WORD_32_BITS) {
    rows[i+0][word_ind] = _mm256_extract_epi32(words, 0);
    rows[i+1][word_ind] = _mm256_extract_epi32(words, 1);
    rows[i+2][word_ind] = _mm256_extract_epi32(words, 2);
    rows[i+3][word_ind] = _mm256_extract_epi32(words, 3);
    rows[i+4][word_ind] = _mm256_extract_epi32(words, 4);
    rows[i+5][word_ind] = _mm256_extract_epi32(words, 5);
    rows[i+6][word_ind] = _mm256_extract_epi32(words, 6);
    rows[i+7][word_ind] = _mm256_extract_epi32(words, 7);
  } else if constexpr (WORD_16_BITS) {
    rows[i+0 ][word_ind] = _mm256_extract_epi16(words, 0 );
    rows[i+1 ][word_ind] = _mm256_extract_epi16(words, 1 );
    rows[i+2 ][word_ind] = _mm256_extract_epi16(words, 2 );
    rows[i+3 ][word_ind] = _mm256_extract_epi16(words, 3 );
    rows[i+4 ][word_ind] = _mm256_extract_epi16(words, 4 );
    rows[i+5 ][word_ind] = _mm256_extract_epi16(words, 5 );
    rows[i+6 ][word_ind] = _mm256_extract_epi16(words, 6 );
    rows[i+7 ][word_ind] = _mm256_extract_epi16(words, 7 );
    rows[i+8 ][word_ind] = _mm256_extract_epi16(words, 8 );
    rows[i+9 ][word_ind] = _mm256_extract_epi16(words, 9 );
    rows[i+10][word_ind] = _mm256_extract_epi16(words, 10);
    rows[i+11][word_ind] = _mm256_extract_epi16(words, 11);
    rows[i+12][word_ind] = _mm256_extract_epi16(words, 12);
    rows[i+13][word_ind] = _mm256_extract_epi16(words, 13);
    rows[i+14][word_ind] = _mm256_extract_epi16(words, 14);
    rows[i+15][word_ind] = _mm256_extract_epi16(words, 15);
  }
}

inline void xor_phase_bits(size_t word_ind, size_t offset, binary_word* phase, __m256i bits) {
  if constexpr (WORD_64_BITS) {
    phase[word_ind] ^= _mm256_extract_epi64(bits, 0) << (offset + 0);
    phase[word_ind] ^= _mm256_extract_epi64(bits, 1) << (offset + 1);
    phase[word_ind] ^= _mm256_extract_epi64(bits, 2) << (offset + 2);
    phase[word_ind] ^= _mm256_extract_epi64(bits, 3) << (offset + 3);
  } else if constexpr (WORD_32_BITS) {
    phase[word_ind] ^= _mm256_extract_epi32(bits, 0) << (offset + 0);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 1) << (offset + 1);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 2) << (offset + 2);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 3) << (offset + 3);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 4) << (offset + 4);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 5) << (offset + 5);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 6) << (offset + 6);
    phase[word_ind] ^= _mm256_extract_epi32(bits, 7) << (offset + 7);
  } else if constexpr (WORD_16_BITS) {
    phase[word_ind] ^= _mm256_extract_epi16(bits, 0 ) << (offset + 0);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 1 ) << (offset + 1);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 2 ) << (offset + 2);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 3 ) << (offset + 3);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 4 ) << (offset + 4);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 5 ) << (offset + 5);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 6 ) << (offset + 6);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 7 ) << (offset + 7);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 8 ) << (offset + 8);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 9 ) << (offset + 9);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 10) << (offset + 10);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 11) << (offset + 11);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 12) << (offset + 12);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 13) << (offset + 13);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 14) << (offset + 14);
    phase[word_ind] ^= _mm256_extract_epi16(bits, 15) << (offset + 15);
  }
}

void TableauSIMD::h(uint32_t a) {
  validate_qubit(a);

  size_t k = 2*num_qubits - (2*num_qubits) % LANES;
  size_t word_ind = (2*a) / binary_word_size();
  size_t offset = (2*a) % binary_word_size();

  __m256i mask_x = set1(1ULL << offset);
  __m256i mask_z = set1(1ULL << (offset+1));

  for (size_t i = 0; i < k; i += LANES) {
    __m256i v = load_words(i, word_ind, rows);

    __m256i bit_x = _mm256_and_si256(v, mask_x);
    __m256i bit_z = _mm256_and_si256(v, mask_z);

    __m256i bit_x_to_z = slli(bit_x, 1);
    __m256i bit_z_to_x = srli(bit_z, 1);

    __m256i cleared = _mm256_andnot_si256(_mm256_or_si256(mask_x, mask_z), v);
    save_words(i, word_ind, rows, _mm256_or_si256(cleared, _mm256_or_si256(bit_x_to_z, bit_z_to_x)));

    __m256i x0 = srli(bit_x, offset);
    __m256i z0 = srli(bit_z, offset+1);
    __m256i x_and_z = _mm256_and_si256(x0, z0);
    xor_phase_bits(i / binary_word_size(), i % binary_word_size(), phase, x_and_z);
  }

  // Scalar version for remainder
  for (size_t i = k; i < 2*num_qubits; i++) {
    constexpr bool h_phase_lookup[] = {0, 0, 0, 1};
    uint8_t xza = get_xz(i, a);
    bool xa = (xza >> 0u) & 1u;
    bool za = (xza >> 1u) & 1u;

    _set(phase, i, _get(phase, i) != h_phase_lookup[xza]);
    set(i, 2*a,   za);
    set(i, 2*a+1, xa);
  }
}

void TableauSIMD::s(uint32_t a) {
  validate_qubit(a);

  size_t k = 2*num_qubits - (2*num_qubits) % LANES;
  size_t word_ind = (2*a) / binary_word_size();
  size_t offset = (2*a) % binary_word_size();

  __m256i mask_x = set1(1ULL << offset);
  __m256i mask_z = set1(1ULL << (offset+1));

  for (size_t i = 0; i < k; i += LANES) {
    __m256i v = load_words(i, word_ind, rows);

    __m256i bit_x = _mm256_and_si256(v, mask_x);
    __m256i bit_z = _mm256_and_si256(v, mask_z);

    __m256i bit_x_to_z = slli(bit_x, 1);

    __m256i cleared = _mm256_andnot_si256(mask_z, v);
    save_words(i, word_ind, rows, _mm256_xor_si256(cleared, _mm256_xor_si256(bit_z, bit_x_to_z)));

    __m256i x0 = srli(bit_x, offset);
    __m256i z0 = srli(bit_z, offset+1);
    __m256i x_and_z = _mm256_and_si256(x0, z0);
    xor_phase_bits(i / binary_word_size(), i % binary_word_size(), phase, x_and_z);
  }

  for (size_t i = k; i < 2*num_qubits; i++) {
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

  size_t k = 2*num_qubits - (2*num_qubits) % LANES;
  size_t word_ind_a = (2*a) / binary_word_size();
  size_t offset_a = (2*a) % binary_word_size();
  __m256i mask_x_a = set1(1ULL << offset_a);
  __m256i mask_z_a = set1(1ULL << (offset_a + 1));

  size_t word_ind_b = (2*b) / binary_word_size();
  size_t offset_b = (2*b) % binary_word_size();
  __m256i mask_x_b = set1(1ULL << offset_b);
  __m256i mask_z_b = set1(1ULL << (offset_b + 1));

  for (size_t i = 0; i < k; i += LANES) {
    __m256i v_a = load_words(i, word_ind_a, rows);
    __m256i bit_x_a = _mm256_and_si256(v_a, mask_x_a);
    __m256i bit_z_a = _mm256_and_si256(v_a, mask_z_a);

    __m256i v_b = load_words(i, word_ind_b, rows);
    __m256i bit_x_b = _mm256_and_si256(v_b, mask_x_b);
    __m256i bit_z_b = _mm256_and_si256(v_b, mask_z_b);

    int bit_shift = std::abs(static_cast<int>(offset_a) - static_cast<int>(offset_b));
    bool left = offset_b < offset_a;

    // update za -> za != zb
    __m256i bit_shifted_zb = left ? slli(bit_z_b, bit_shift) : srli(bit_z_b, bit_shift);
    __m256i cleared_a = _mm256_andnot_si256(mask_z_a, v_a);
    __m256i v_new_a = _mm256_or_si256(cleared_a, _mm256_xor_si256(bit_shifted_zb, bit_z_a));

    if (word_ind_a == word_ind_b) {
      v_b = v_new_a;
    } else {
      save_words(i, word_ind_a, rows, v_new_a);
    }

    // update xb -> xa != xb
    __m256i bit_shifted_xa = left ? srli(bit_x_a, bit_shift) : slli(bit_x_a, bit_shift);
    __m256i cleared_b = _mm256_andnot_si256(mask_x_b, v_b);
    save_words(i, word_ind_b, rows, _mm256_or_si256(cleared_b, _mm256_xor_si256(bit_shifted_xa, bit_x_b)));


    __m256i x0_a = srli(bit_x_a, offset_a);
    __m256i z0_a = srli(bit_z_a, offset_a+1);
    __m256i x0_b = srli(bit_x_b, offset_b);
    __m256i z0_b = srli(bit_z_b, offset_b+1);
    __m256i one_vec = set1(1);
    __m256i result = _mm256_and_si256(_mm256_and_si256(x0_a, z0_b), _mm256_xor_si256(_mm256_xor_si256(x0_b, z0_a), one_vec));
    xor_phase_bits(i / binary_word_size(), i % binary_word_size(), phase, result);
  }

  for (size_t i = k; i < 2*num_qubits; i++) {
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

void TableauSIMD::swap(size_t i, size_t j) {
  std::swap(rows[i], rows[j]);
  bool r1 = _get(phase, i);
  bool r2 = _get(phase, j);
  _set(phase, i, r2);
  _set(phase, j, r1);
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

    swap(p, p - num_qubits);


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
#else
// Dummy implementation for when AVX2 is not available
TableauSIMD::TableauSIMD(uint32_t num_qubits) {
  throw std::runtime_error("Cannot create a SIMD Tableau without AVX2 instructions.");
}
Pauli TableauSIMD::get_pauli(size_t i, size_t j) const {}
PauliString TableauSIMD::get_stabilizer(size_t i) const {}
PauliString TableauSIMD::get_destabilizer(size_t i) const {}
uint8_t TableauSIMD::get_phase(size_t i) const {}
void TableauSIMD::reset(int i) {}
void TableauSIMD::rowsum(int i, int j) {}
TableauSIMD::~TableauSIMD() {}
bool TableauSIMD::operator==(TableauSIMD& other) {}
void TableauSIMD::rref(const Qubits& sites) {}
void TableauSIMD::xrref(const Qubits& sites) {}
uint32_t TableauSIMD::xrank(const Qubits& sites) {}
uint32_t TableauSIMD::rank(const Qubits& sites) {}
void TableauSIMD::rref() {}
void TableauSIMD::xrref() {}
uint32_t TableauSIMD::rank() {}
uint32_t TableauSIMD::xrank() {}
double TableauSIMD::bitstring_amplitude(const BitString& bits) {}
std::string TableauSIMD::to_string(bool print_destabilizers) const {}
std::string TableauSIMD::to_string_ops(bool print_destabilizers) const {}
void TableauSIMD::h(uint32_t a) {}
void TableauSIMD::s(uint32_t a) {}
void TableauSIMD::cx(uint32_t a, uint32_t b) {}
std::pair<bool, uint32_t> TableauSIMD::mzr_deterministic(uint32_t a) const {}
void TableauSIMD::swap(size_t i, size_t j) {}
MeasurementData TableauSIMD::mzr(uint32_t a, std::optional<bool> outcome) {}
#endif
