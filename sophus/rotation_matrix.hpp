/// @file
/// Rotation matrix helper functions.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "types.hpp"

namespace sophus {

/// Takes in arbitrary square matrix and returns true if it is
/// orthogonal.
template <class D>
SOPHUS_FUNC bool isOrthogonal(Eigen::MatrixBase<D> const& R) {
  using Scalar = typename D::Scalar;
  static int const kMatrixDim = D::RowsAtCompileTime;
  static int const M = D::ColsAtCompileTime;

  static_assert(kMatrixDim == M, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  return (R * R.transpose() - Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity()).norm() <
         Constants<Scalar>::epsilon();
}

/// Takes in arbitrary square matrix and returns true if it is
/// "scaled-orthogonal" with positive determinant.
///
template <class D>
SOPHUS_FUNC bool isScaledOrthogonalAndPositive(Eigen::MatrixBase<D> const& sR) {
  using Scalar = typename D::Scalar;
  static int const kMatrixDim = D::RowsAtCompileTime;
  static int const M = D::ColsAtCompileTime;
  using std::pow;
  using std::sqrt;

  Scalar det = sR.determinant();

  if (det <= Scalar(0)) {
    return false;
  }

  Scalar scale_sqr = pow(det, Scalar(2. / kMatrixDim));

  static_assert(kMatrixDim == M, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  return (sR * sR.transpose() - scale_sqr * Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity())
             .template lpNorm<Eigen::Infinity>() <
         sqrt(Constants<Scalar>::epsilon());
}

/// Takes in arbitrary square matrix (2x2 or larger) and returns closest
/// orthogonal matrix with positive determinant.
template <class D>
SOPHUS_FUNC enable_if_t<
    std::is_floating_point<typename D::Scalar>::value,
    Matrix<typename D::Scalar, D::RowsAtCompileTime, D::RowsAtCompileTime>>
makeRotationMatrix(Eigen::MatrixBase<D> const& R) {
  using Scalar = typename D::Scalar;
  static int const kMatrixDim = D::RowsAtCompileTime;
  static int const M = D::ColsAtCompileTime;

  static_assert(kMatrixDim == M, "must be a square matrix");
  static_assert(kMatrixDim >= 2, "must have compile time dimension >= 2");

  Eigen::JacobiSVD<Matrix<Scalar, kMatrixDim, kMatrixDim>> svd(
      R, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Determine determinant of orthogonal matrix U*V'.
  Scalar d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  // Starting from the identity matrix D, set the last entry to d (+1 or
  // -1),  so that det(U*D*V') = 1.
  Matrix<Scalar, kMatrixDim, kMatrixDim> Diag = Matrix<Scalar, kMatrixDim, kMatrixDim>::Identity();
  Diag(kMatrixDim - 1, kMatrixDim - 1) = d;
  return svd.matrixU() * Diag * svd.matrixV().transpose();
}

}  // namespace sophus
