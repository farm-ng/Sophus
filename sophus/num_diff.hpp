/// @file
/// Numerical differentiation using finite differences

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include "types.hpp"

namespace sophus {

namespace details {
template <class Scalar>
class Curve {
 public:
  template <class Fn>
  static auto num_diff(Fn curve, Scalar t, Scalar h) -> decltype(curve(t)) {
    using ReturnType = decltype(curve(t));
    static_assert(std::is_floating_point<Scalar>::value,
                  "Scalar must be a floating point type.");
    static_assert(IsFloatingPoint<ReturnType>::value,
                  "ReturnType must be either a floating point scalar, "
                  "vector or matrix.");

    return (curve(t + h) - curve(t - h)) / (Scalar(2) * h);
  }
};

template <class Scalar, int kMatrixDim, int M>
class VectorField {
 public:
  static Eigen::Matrix<Scalar, kMatrixDim, M> num_diff(
      std::function<sophus::Vector<Scalar, kMatrixDim>(sophus::Vector<Scalar, M>)>
          vector_field,
      sophus::Vector<Scalar, M> const& a, Scalar eps) {
    static_assert(std::is_floating_point<Scalar>::value,
                  "Scalar must be a floating point type.");
    Eigen::Matrix<Scalar, kMatrixDim, M> J;
    sophus::Vector<Scalar, M> h;
    h.setZero();
    for (int i = 0; i < M; ++i) {
      h[i] = eps;
      J.col(i) =
          (vector_field(a + h) - vector_field(a - h)) / (Scalar(2) * eps);
      h[i] = Scalar(0);
    }

    return J;
  }
};

template <class Scalar, int kMatrixDim>
class VectorField<Scalar, kMatrixDim, 1> {
 public:
  static Eigen::Matrix<Scalar, kMatrixDim, 1> num_diff(
      std::function<sophus::Vector<Scalar, kMatrixDim>(Scalar)> vector_field,
      Scalar const& a, Scalar eps) {
    return details::Curve<Scalar>::num_diff(std::move(vector_field), a, eps);
  }
};
}  // namespace details

/// Calculates the derivative of a curve at a point ``t``.
///
/// Here, a curve is a function from a Scalar to a Euclidean space. Thus, it
/// returns either a Scalar, a vector or a matrix.
///
template <class Scalar, class Fn>
auto curveNumDiff(Fn curve, Scalar t,
                  Scalar h = Constants<Scalar>::epsilonSqrt())
    -> decltype(details::Curve<Scalar>::num_diff(std::move(curve), t, h)) {
  return details::Curve<Scalar>::num_diff(std::move(curve), t, h);
}

/// Calculates the derivative of a vector field at a point ``a``.
///
/// Here, a vector field is a function from a vector space to another vector
/// space.
///
template <class Scalar, int kMatrixDim, int M, class ScalarOrVector, class Fn>
Eigen::Matrix<Scalar, kMatrixDim, M> vectorFieldNumDiff(
    Fn vector_field, ScalarOrVector const& a,
    Scalar eps = Constants<Scalar>::epsilonSqrt()) {
  return details::VectorField<Scalar, kMatrixDim, M>::num_diff(std::move(vector_field),
                                                      a, eps);
}

}  // namespace sophus
