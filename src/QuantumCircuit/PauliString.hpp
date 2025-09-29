#pragma once

#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <ranges>

#include "CircuitUtils.h"
#include "Random.hpp"

class QuantumCircuit;

template <typename T>
static void remove_even_indices(std::vector<T> &v) {
  uint32_t vlen = v.size();
  for (uint32_t i = 0; i < vlen; i++) {
    uint32_t j = vlen - i - 1;
    if ((j % 2)) {
      v.erase(v.begin() + j);
    }
  }
}

enum Pauli {
  I, X, Z, Y
};

constexpr int compute_phase(uint8_t xz1, uint8_t xz2) {
  bool x1 = (xz1 >> 0u) & 1u;
  bool z1 = (xz1 >> 1u) & 1u;
  bool x2 = (xz2 >> 0u) & 1u;
  bool z2 = (xz2 >> 1u) & 1u;
  if (!x1 && !z1) { 
    return 0; 
  } else if (x1 && z1) {
    if (z2) { 
      return x2 ? 0 : 1;
    } else { 
      return x2 ? -1 : 0;
    }
  } else if (x1 && !z1) {
    if (z2) { 
      return x2 ? 1 : -1;
    } else { 
      return 0; 
    }
  } else {
    if (x2) {
      return z2 ? -1 : 1;
    } else { 
      return 0; 
    }
  }
}

constexpr std::array<int, 16> generate_phase_table() {
  std::array<int, 16> table;
  for (uint8_t xz1 = 0; xz1 < 4; xz1++) {
    for (uint8_t xz2 = 0; xz2 < 4; xz2++) {
      table[xz2 + (xz1 << 2u)] = compute_phase(xz1, xz2);
    }
  }

  return table;
}

constexpr int multiplication_phase(uint8_t xz1, uint8_t xz2) {
  constexpr auto results = generate_phase_table();
  return results[xz2 + (xz1 << 2u)];
}

constexpr std::pair<Pauli, uint8_t> multiply_pauli(Pauli p1, Pauli p2) {
  uint8_t p1_bits = static_cast<uint8_t>(p1);
  uint8_t p2_bits = static_cast<uint8_t>(p2);

  int phase = multiplication_phase(p1_bits, p2_bits);
  constexpr uint8_t phase_bits[] = {0b11, 0b00, 0b01};

  return {static_cast<Pauli>(p1_bits ^ p2_bits), phase_bits[phase + 1]};
}

constexpr char pauli_to_char(Pauli p) {
  if (p == Pauli::I) {
    return 'I';
  } else if (p == Pauli::X) {
    return 'X';
  } else if (p == Pauli::Y) {
    return 'Y';
  } else if (p == Pauli::Z) {
    return 'Z';
  }
}

constexpr std::complex<double> sign_from_bits(uint8_t phase) {
  constexpr std::complex<double> i(0.0, 1.0);
  if (phase == 0) {
    return 1.0;
  } else if (phase == 1) {
    return i;
  } else if (phase == 2) {
    return -1.0;
  } else {
    return -i;
  }
}


using binary_word = uint64_t;

static inline constexpr size_t binary_word_size() {
  return 8u*sizeof(binary_word);
}

struct BitString {
  uint32_t num_bits;
  std::vector<binary_word> bits;

  BitString()=default;

  BitString(uint32_t num_bits);

  binary_word to_integer() const;

  static BitString from_bits(size_t num_bits, binary_word bits);

  static BitString random(size_t num_bits, double p = 0.5);

  uint32_t hamming_weight() const;

  QubitInterval support_range() const;

  inline binary_word get(uint32_t i) const {
    binary_word word = bits[i / binary_word_size()];
    uint32_t bit_ind = i % binary_word_size();

    return (word >> bit_ind) & static_cast<binary_word>(1);
  }

  inline void set(uint32_t i, binary_word v) {
    uint32_t word_ind = i / binary_word_size();
    uint32_t bit_ind = i % binary_word_size();

    bits[word_ind] = (bits[word_ind] & ~(static_cast<binary_word>(1) << bit_ind)) | (v << bit_ind);
  }

  uint32_t size() const;

  const binary_word& operator[](uint32_t i) const;

  binary_word& operator[](uint32_t i);

  BitString operator^(const BitString& other) const;

  BitString& operator^=(const BitString& other);

  BitString substring(const std::vector<uint32_t>& kept_bits, bool remove_bits=false) const;

  BitString superstring(const std::vector<uint32_t>& sites, size_t new_num_bits) const;
};

class PauliString {
  public:
    uint32_t num_qubits;
    uint8_t phase;

    // Store bitstring as an array of 32-bit words
    // The bits are formatted as:
    // x0 z0 x1 z1 ... x15 z15
    // x16 z16 x17 z17 ... etc
    // This appears to be slightly more efficient than the originally format originally described
    // by Aaronson and Gottesman (https://arxiv.org/abs/quant-ph/0406196) as it 
    // is more cache-friendly; most operations only act on a single word.
    BitString bit_string;

    PauliString()=default;
    PauliString(uint32_t num_qubits);
    PauliString(const PauliString& other);

    static uint32_t process_pauli_string(const std::string& paulis);

    static inline uint8_t parse_phase(std::string& s) {
      if (s.rfind("+", 0) == 0) {
        if (s.rfind("+i", 0) == 0) {
          s = s.substr(2);
          return 1;
        } else {
          s = s.substr(1);
          return 0;
        }
      } else if (s.rfind("-", 0) == 0) {
        if (s.rfind("-i", 0) == 0) {
          s = s.substr(2);
          return 3;
        } else {
          s = s.substr(1);
          return 2;
        }
      }

      return 0;
    }

    PauliString(const std::string& paulis);

    PauliString(const std::vector<Pauli>& paulis, uint8_t phase=0);

    static PauliString rand(uint32_t num_qubits);

    static PauliString randh(uint32_t num_qubits);

    static PauliString basis(uint32_t num_qubits, const std::string& P, uint32_t q, uint8_t r);

    static PauliString from_bitstring(uint32_t num_qubits, uint32_t bits);

    static PauliString basis(uint32_t num_qubits, const std::string& P, uint32_t q);

    PauliString substring(const Qubits& qubits, bool remove_qubits=false) const;

    PauliString substring(const QubitSupport& support, bool remove_qubits=false) const;

    PauliString superstring(const Qubits& qubits, size_t new_num_qubits) const;

    static uint8_t get_multiplication_phase(const PauliString& p1, const PauliString& p2);

    bool hermitian() const;
    bool is_basis() const;

    PauliString operator*(const PauliString& other) const;
    PauliString operator-();
    bool operator==(const PauliString &rhs) const;
    bool operator!=(const PauliString &rhs) const;

    friend std::ostream& operator<< (std::ostream& stream, const PauliString& p) {
      stream << p.to_string_ops();
      return stream;
    }


    Eigen::Matrix2cd to_matrix(uint32_t i) const;
    Eigen::MatrixXcd to_matrix() const;

    Pauli to_pauli(uint32_t i) const;
    std::vector<Pauli> to_pauli() const;

    QubitInterval support_range() const;
    Qubits get_support() const;

    std::string to_op(uint32_t i) const;

    inline static std::string phase_to_string(uint8_t phase) {
      if (phase == 0) {
        return "+";
      } else if (phase == 1) {
        return "+i";
      } else if (phase == 2) {
        return "-";
      } else if (phase == 3) {
        return "-i";
      }

      throw std::runtime_error("Invalid phase bits passed to phase_to_string.");
    }

    std::string to_string() const;
    std::string to_string_ops() const;

    void evolve(const QuantumCircuit& qc);

    void s(uint32_t a);
    void sd(uint32_t a);
    void h(uint32_t a);
    void x(uint32_t a);
    void y(uint32_t a);
    void z(uint32_t a);
    void sqrtX(uint32_t a);
    void sqrtXd(uint32_t a);
    void sqrtY(uint32_t a);
    void sqrtYd(uint32_t a);
    void sqrtZ(uint32_t a);
    void sqrtZd(uint32_t a);
    void cx(uint32_t a, uint32_t b);
    void cy(uint32_t a, uint32_t b);
    void cz(uint32_t a, uint32_t b);
    void swap(uint32_t a, uint32_t b);

    bool commutes_at(const PauliString& p, uint32_t i) const;
    bool commutes(const PauliString& p) const;

    template <typename... Args>
    void reduce(bool to_z, Args... args) const {
      PauliString p(*this);
      p.reduce_inplace(to_z, args...);
    }

    template <typename... Args>
    void reduce_inplace(bool to_z, Args... args) {
      if (to_z) {
        h(0);
        (args.first->h(args.second[0]), ...);
      }

      // Step one
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (get_z(i)) {
          if (get_x(i)) {
            s(i);
            (args.first->s(args.second[i]), ...);
          } else {
            h(i);
            (args.first->h(args.second[i]), ...);
          }
        }
      }

      // Step two
      std::vector<uint32_t> nonzero_idx;
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (get_x(i)) {
          nonzero_idx.push_back(i);
        }
      }

      while (nonzero_idx.size() > 1) {
        for (uint32_t j = 0; j < nonzero_idx.size()/2; j++) {
          uint32_t q1 = nonzero_idx[2*j];
          uint32_t q2 = nonzero_idx[2*j+1];
          cx(q1, q2);
          (args.first->cx(args.second[q1], args.second[q2]), ...);
        }

        remove_even_indices(nonzero_idx);
      }

      // Step three
      uint32_t ql = nonzero_idx[0];
      if (ql != 0) {
        for (uint32_t i = 0; i < num_qubits; i++) {
          if (get_x(i)) {
            cx(0, ql);
            cx(ql, 0);
            cx(0, ql);

            (args.first->cx(args.second[0], args.second[ql]), ...);
            (args.first->cx(args.second[ql], args.second[0]), ...);
            (args.first->cx(args.second[0], args.second[ql]), ...);

            break;
          }
        }
      }

      if (to_z) {
        h(0);
        (args.first->h(args.second[0]), ...);

        if (phase == 2) {
          x(0);
          (args.first->x(args.second[0]), ...);
        }
      } else {
        if (phase == 2) {
          z(0);
          (args.first->z(args.second[0]), ...);
        }
      }
    }

    // Returns the circuit which maps this PauliString onto p
    QuantumCircuit transform(PauliString const &p) const;

    inline std::complex<double> sign() const {
      return sign_from_bits(phase);
    }

    inline bool get(size_t i) const {
      return bit_string.get(i);
    }

    inline bool get_x(uint32_t i) const {
      return bit_string.get(2*i);
    }

    inline bool get_z(uint32_t i) const {
      return bit_string.get(2*i + 1);
    }

    // It is slightly faster (~20-30%) to query both the x and z bits at a given site
    // at the same time, storing them in the first two bits of the return value.
    inline uint8_t get_xz(uint32_t i) const {
      constexpr uint32_t num_paulis = binary_word_size()/2;
      uint32_t bit_ind = 2u*(i % num_paulis);
      return 0u | (((bit_string.bits[i / num_paulis] >> bit_ind) & 3u) << 0u);
    }

    inline uint8_t get_r() const { 
      return phase; 
    }

    inline void set(size_t i, binary_word v) {
      bit_string.set(i, v);
    }

    inline void set_x(uint32_t i, binary_word v) {
      bit_string.set(2*i, v);
    }

    inline void set_z(uint32_t i, binary_word v) {
      bit_string.set(2*i + 1, v);
    }

    inline void set_r(uint8_t v) { 
      phase = v & 0b11; 
    }

    inline void set_op(size_t i, Pauli p) {
      if (p == Pauli::I) {
        set_x(i, false);
        set_z(i, false);
      } else if (p == Pauli::X) {
        set_x(i, true);
        set_z(i, false);
      } else if (p == Pauli::Y) {
        set_x(i, true);
        set_z(i, true);
      } else if (p == Pauli::Z) {
        set_x(i, false);
        set_z(i, true);
      }
    }
};

using SparsePauliObs = std::vector<std::tuple<std::complex<double>, PauliString, Qubits>>;

namespace fmt {
  template <>
  struct formatter<BitString> {
    std::optional<size_t> width;

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
      auto it = ctx.begin(), end = ctx.end();

      if (it != end && *it >= '0' && *it <= '9') {
        size_t k = 0;
        while (it != end && *it >= '0' && *it <= '9') {
          k = k * 10 + (*it - '0');
          ++it;
        }
        width = k;
      }

      return it;
    }

    template <typename FormatContext>
    auto format(const BitString& bs, FormatContext& ctx) const -> decltype(ctx.out()) {
      std::string bit_str = "";
      for (size_t i = 0; i < bs.size(); i++) {
        bit_str += fmt::format("{:032b}", bs.bits[i]);
      }

      size_t n = bit_str.size();
      size_t k = width ? width.value() : bs.num_bits;
      bit_str = bit_str.substr(n - k, n);

      if (width && width.value() > bit_str.size()) {
        bit_str.insert(0, width.value() - bit_str.size(), '0');
      }

      return fmt::format_to(ctx.out(), "{}", bit_str);
    }
  };
}

namespace fmt {
  template <>
  struct formatter<PauliString> {
    constexpr auto parse(format_parse_context& ctx) const -> decltype(ctx.begin()) {
      return ctx.begin();
    }

    // Format function
    template <typename FormatContext>
      auto format(const PauliString& ps, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", ps.to_string_ops());
      }
  };
}
