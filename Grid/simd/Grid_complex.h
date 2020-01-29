/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/Grid_complex.h

Copyright (C) 2015

Author: Gianluca Filaci <g.filaci@ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */

NAMESPACE_BEGIN(Grid);

template<class T> class complex;

template<class T> constexpr complex<T> operator+(const complex<T>& z, const complex<T>& w) {
  complex<T> r = z;
  r += w;
  return r;
}
template<class T> constexpr complex<T> operator+(const complex<T>& z, const T& w) {
  complex<T> r = z;
  r += w;
  return r;
}
template<class T> constexpr complex<T> operator+(const T& z, const complex<T>& w) {
  complex<T> r = w;
  r += z;
  return r;
}
template<class T> constexpr complex<T> operator-(const complex<T>& z, const complex<T>& w) {
  complex<T> r = z;
  r -= w;
  return r;
}
template<class T> constexpr complex<T> operator-(const complex<T>& z, const T& w) {
  complex<T> r = z;
  r -= w;
  return r;
}
template<class T> constexpr complex<T> operator-(const T& z, const complex<T>& w) {
  complex<T> r = w;
  r -= z;
  return r;
}
template<class T> constexpr complex<T> operator*(const complex<T>& z, const complex<T>& w) {
  complex<T> r = z;
  r *= w;
  return r;
}
template<class T> constexpr complex<T> operator*(const complex<T>& z, const T& w) {
  complex<T> r = z;
  r *= w;
  return r;
}
template<class T> constexpr complex<T> operator*(const T& z, const complex<T>& w) {
  complex<T> r = w;
  r *= z;
  return r;
}
template<class T> constexpr complex<T> operator/(const complex<T>& z, const complex<T>& w) {
  complex<T> r = z;
  r /= w;
  return r;
}
template<class T> constexpr complex<T> operator/(const complex<T>& z, const T& w) {
  complex<T> r = z;
  r /= w;
  return r;
}
template<class T> constexpr complex<T> operator/(const T& z, const complex<T>& w) {
  complex<T> r = z;
  r /= w;
  return r;
}

template<class T> constexpr complex<T> operator+(const complex<T>& z) { return z; };
template<class T> constexpr complex<T> operator-(const complex<T>& z) { return complex<T>(-z.real(), -z.imag()); }

template<class T> constexpr bool operator==(const complex<T>& z, const complex<T>& w) { return z.real() == w.real() && z.imag() == w.imag(); }
template<class T> constexpr bool operator==(const complex<T>& z, const T& w) { return z.real() == w && z.imag() == T(); }
template<class T> constexpr bool operator==(const T& z, const complex<T>& w) { return z == w.real() && w.imag() == T(); }
template<class T> constexpr bool operator!=(const complex<T>& z, const complex<T>& w) { return z.real() != w.real() || z.imag() != w.imag(); }
template<class T> constexpr bool operator!=(const complex<T>& z, const T& w) { return z.real() != w || z.imag() != T(); }
template<class T> constexpr bool operator!=(const T& z, const complex<T>& w) { return z != w.real() || w.imag() != T(); }

template<class T, class charT, class traits>
  std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const complex<T>& z) {
  stream << "(" << z.real() << "," << z.imag() << ")";
  return stream;
}

template<class T> constexpr T real(const complex<T>& z) { return z.real(); }
template<class T> constexpr T imag(const complex<T>& z) { return z.imag(); }
template<class T> inline T abs(const complex<T>& z) {
  T re = z.real();
  T im = z.imag();
  const T s = cl::sycl::max(cl::sycl::abs(re), cl::sycl::abs(im));
  if (s == T()) return s;
  re /= s;
  im /= s;
  return s * cl::sycl::sqrt(re * re + im * im);
}
template<class T> inline T arg(const complex<T>& z) { return  cl::sycl::atan2(z.imag(), z.real()); }
template<class T> constexpr T norm(const complex<T>& z) { const T re = z.real(); const T im = z.imag(); return re * re + im * im; }
template<class T> constexpr complex<T> conj(const complex<T>& z) { return complex<T>(z.real(), -z.imag()); }
template<class T> inline complex<T> polar(const T& rho, const T& theta = T()) { return complex<T>(rho * cl::sycl::cos(theta), rho * cl::sycl::sin(theta)); }
template<class T> inline complex<T> exp  (const complex<T>& z) { return Grid::polar<T>(cl::sycl::exp(z.real()), z.imag()); };
template<class T> inline complex<T> log  (const complex<T>& z) { return complex<T>(cl::sycl::log(abs(z)), arg(z)); }
template<class T> inline complex<T> pow  (const complex<T>& z, const T& n) {
  complex<T> y = n % 2 ? z : complex<T>(1);
  while (n >>= 1) {
    z *= z;
    if (n % 2) y *= z;
  }
  return y;
}
template<class T> inline complex<T> sin  (const complex<T>& z) {
  return complex<T>(cl::sycl::sin(z.x) * cl::sycl::cosh(z.y), cl::sycl::cos(z.x) * cl::sycl::sinh(z.y));
}


// ==================================================================== //
// these can be implemented later...

template<class T, class charT, class traits>
  std::basic_istream<charT, traits>& operator>>(std::basic_istream<charT, traits>&, complex<T>&);

template<class T> inline complex<T> proj(const complex<T>&);
template<class T> inline complex<T> acos(const complex<T>&);
template<class T> inline complex<T> asin(const complex<T>&);
template<class T> inline complex<T> atan(const complex<T>&);
template<class T> inline complex<T> acosh(const complex<T>&);
template<class T> inline complex<T> asinh(const complex<T>&);
template<class T> inline complex<T> atanh(const complex<T>&);
template<class T> inline complex<T> cos  (const complex<T>&);
template<class T> inline complex<T> cosh (const complex<T>&);
template<class T> inline complex<T> log10(const complex<T>&);
template<class T> inline complex<T> sinh (const complex<T>&);
template<class T> inline complex<T> tan  (const complex<T>&);
template<class T> inline complex<T> tanh (const complex<T>&);
template<class T> inline complex<T> pow  (const complex<T>&, const complex<T>&);
template<class T> inline complex<T> pow  (const T&, const complex<T>&);
template<class T> inline complex<T> sqrt (const complex<T>&);

inline namespace literals {
  inline namespace complex_literals {
    constexpr complex<long double> operator""_il(long double);
    constexpr complex<long double> operator""_il(unsigned long long);
    constexpr complex<double> operator""_i(long double);
    constexpr complex<double> operator""_i(unsigned long long);
    constexpr complex<float> operator""_if(long double);
    constexpr complex<float> operator""_if(unsigned long long);
  }
}
// ==================================================================== //

template<class T> class complex {
 private:
  T x, y;

 public:
  using value_type = T;

  constexpr complex(const T& re = T(), const T& im = T()) : x(re), y(im) {}
  constexpr complex(const complex&) = default;
  template<class X> constexpr complex(const complex<X>& z) : x(z.real()), y(z.imag()) {}

  constexpr complex(const std::complex<T>& z) : x(z.real()), y(z.imag()) {}
  constexpr operator std::complex<T>() const { return std::complex<T>(x, y); }
  template<class X> constexpr complex& operator= (const std::complex<X>& z) { *this  = complex<X>(z); return *this; }
  template<class X> constexpr complex& operator+=(const std::complex<X>& z) { *this += complex<X>(z); return *this; }
  template<class X> constexpr complex& operator-=(const std::complex<X>& z) { *this -= complex<X>(z); return *this; }
  template<class X> constexpr complex& operator*=(const std::complex<X>& z) { *this *= complex<X>(z); return *this; }
  template<class X> constexpr complex& operator/=(const std::complex<X>& z) { *this /= complex<X>(z); return *this; }

  constexpr T real() const { return x; }
  constexpr T imag() const { return y; }
  constexpr void real(T val) { x = val; }
  constexpr void imag(T val) { y = val; }

  constexpr complex& operator= (const T& val) { x  = val; y  = T(); return *this; }
  constexpr complex& operator+=(const T& val) { x += val; return *this; }
  constexpr complex& operator-=(const T& val) { x -= val; return *this; }
  constexpr complex& operator*=(const T& val) { x *= val; y *= val; return *this; }
  constexpr complex& operator/=(const T& val) { x /= val; y /= val; return *this; }
  constexpr complex& operator=(const complex& z) { x = z.real(); y = z.imag(); return *this; }

  template<class X> constexpr complex& operator= (const complex<X>& z) { x  = z.real(); y  = z.imag(); return *this; }
  template<class X> constexpr complex& operator+=(const complex<X>& z) { x += z.real(); y += z.imag(); return *this; }
  template<class X> constexpr complex& operator-=(const complex<X>& z) { x -= z.real(); y -= z.imag(); return *this; }
  template<class X> constexpr complex& operator*=(const complex<X>& z) {
    const T r = x * z.real() - y * z.imag();
    y = x * z.imag() + y * z.real();
    x = r;
    return *this;
  }
  template<class X> constexpr complex& operator/=(const complex<X>& z) {
    const T r = x * z.real() + y * z.imag();
    const T n = Grid::norm(z);
    y = (y * z.real() - x * z.imag()) / n;
    x = r / n;
    return *this;
  }
};

NAMESPACE_END(Grid);
