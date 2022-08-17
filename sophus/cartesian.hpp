/// @file
/// Cartesian - Euclidean vector space as Lie group

#pragma once
#include <sophus/types.hpp>

namespace sophus {
template <class ScalarT, int M, int Options = 0>
class Cartesian;

template <class ScalarT>
using Cartesian2 = Cartesian<ScalarT, 2>;

template <class ScalarT>
using Cartesian3 = Cartesian<ScalarT, 3>;

using Cartesian2d = Cartesian2<double>;
using Cartesian3d = Cartesian3<double>;

}  // namespace sophus

namespace Eigen {
namespace internal {

template <class ScalarT, int M, int Options>
struct traits<sophus::Cartesian<ScalarT, M, Options>> {
  using Scalar = ScalarT;
  using ParamsType = sophus::Vector<Scalar, M, Options>;
};

template <class ScalarT, int M, int Options>
struct traits<Map<sophus::Cartesian<ScalarT, M>, Options>>
    : traits<sophus::Cartesian<ScalarT, M, Options>> {
  using Scalar = ScalarT;
  using ParamsType = Map<sophus::Vector<Scalar, M>, Options>;
};

template <class ScalarT, int M, int Options>
struct traits<Map<sophus::Cartesian<ScalarT, M> const, Options>>
    : traits<sophus::Cartesian<ScalarT, M, Options> const> {
  using Scalar = ScalarT;
  using ParamsType = Map<sophus::Vector<Scalar, M> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// Cartesian base type - implements Cartesian class but is storage agnostic.
///
/// Euclidean vector space as Lie group.
///
/// Lie groups can be seen as a generalization over the Euclidean vector
/// space R^M. Here a kMatrixDim-dimensional vector ``p`` is represented as a
//  (M+1) x (M+1) homogeneous matrix:
///
///   | I p |
///   | o 1 |
///
/// On the other hand, Cartesian(M) can be seen as a special case of SE(M)
/// with identity rotation, and hence represents pure translation.
///
/// The purpose of this class is two-fold:
///  - for educational purpose, to highlight how Lie groups generalize over
///    Euclidean vector spaces.
///  - to be used in templated/generic algorithms (such as sophus::Spline)
///    which are implemented against the Lie group interface.
///
/// Obviously, Cartesian(M) can just be represented as a M-tuple.
///
/// Cartesian is not compact, but a commutative group. For vector additions it
/// holds `a+b = b+a`.
///
/// See Cartesian class  for more details below.
///
template <class Derived, int M>
class CartesianBase {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using ParamsType = typename Eigen::internal::traits<Derived>::ParamsType;
  /// Degrees of freedom of manifold, equals to number of Cartesian coordinates.
  static int constexpr kDoF = M;
  /// Number of internal parameters used, also M.
  static int constexpr kNumParameters = M;
  /// Group transformations are (M+1)x(M+1) matrices.
  static int constexpr kMatrixDim = M + 1;
  static int constexpr kPointDim = M;

  using Transformation = sophus::Matrix<Scalar, kMatrixDim, kMatrixDim>;
  using Point = sophus::Vector<Scalar, M>;
  using HomogeneousPoint = sophus::Vector<Scalar, kMatrixDim>;
  using Line = ParametrizedLine<Scalar, M>;
  using Hyperplane = Eigen::Hyperplane<Scalar, M>;
  using Tangent = sophus::Vector<Scalar, kDoF>;
  using Adjoint = Matrix<Scalar, kDoF, kDoF>;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with Cartesian operations.
  template <typename OtherDerived>
  using ReturnScalar = typename Eigen::ScalarBinaryOpTraits<
      Scalar, typename OtherDerived::Scalar>::ReturnType;

  template <typename OtherDerived>
  using CartesianSum = Cartesian<ReturnScalar<OtherDerived>, M>;

  template <typename PointDerived>
  using PointProduct = sophus::Vector<ReturnScalar<PointDerived>, M>;

  template <typename HPointDerived>
  using HomogeneousPointProduct =
      sophus::Vector<ReturnScalar<HPointDerived>, kMatrixDim>;

  /// Adjoint transformation
  ///
  /// Always identity of commutative groups.
  SOPHUS_FUNC Adjoint Adj() const { return Adjoint::Identity(); }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class NewScalarType>
  SOPHUS_FUNC Cartesian<NewScalarType, M> cast() const {
    return Cartesian<NewScalarType, M>(params().template cast<NewScalarType>());
  }

  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  ///
  SOPHUS_FUNC Matrix<Scalar, kNumParameters, kDoF> Dx_this_mul_exp_x_at_0()
      const {
    sophus::Matrix<Scalar, kNumParameters, kDoF> m;
    m.setIdentity();
    return m;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC Matrix<Scalar, kNumParameters, kDoF> Dx_log_this_inv_by_x_at_this()
      const {
    Matrix<Scalar, kDoF, kNumParameters> m;
    m.setIdentity();
    return m;
  }

  /// Returns group inverse.
  ///
  /// The additive inverse.
  ///
  SOPHUS_FUNC Cartesian<Scalar, M> inverse() const {
    return Cartesian<Scalar, M>(-params());
  }

  /// Logarithmic map
  ///
  /// For Euclidean vector space, just the identity. Or to be more precise
  /// it just extracts the significant M-vector from the NxN matrix.
  ///
  SOPHUS_FUNC Tangent log() const { return params(); }

  /// Returns 4x4 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  ///   | I p |
  ///   | o 1 |
  ///
  SOPHUS_FUNC Transformation matrix() const {
    sophus::Matrix<Scalar, kMatrixDim, kMatrixDim> matrix;
    matrix.setIdentity();
    matrix.col(M).template head<M>() = params();
    return matrix;
  }

  /// Group multiplication, are vector additions.
  ///
  template <class OtherDerived>
  SOPHUS_FUNC CartesianBase<Derived, M>& operator=(
      CartesianBase<OtherDerived, M> const& other) {
    params() = other.params();
    return *this;
  }

  /// Group multiplication, are vector additions.
  ///
  template <typename OtherDerived>
  SOPHUS_FUNC CartesianSum<OtherDerived> operator*(
      CartesianBase<OtherDerived, M> const& other) const {
    return CartesianSum<OtherDerived>(params() + other.params());
  }

  /// Group action on points, again just vector addition.
  ///
  template <typename PointDerived,
            typename = typename std::enable_if<
                IsFixedSizeVector<PointDerived, M>::value>::type>
  SOPHUS_FUNC PointProduct<PointDerived> operator*(
      Eigen::MatrixBase<PointDerived> const& p) const {
    return PointProduct<PointDerived>(params() + p);
  }

  /// Group action on homogeneous points. See above for more details.
  ///
  template <typename HPointDerived,
            typename = typename std::enable_if<
                IsFixedSizeVector<HPointDerived, kMatrixDim>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<HPointDerived> operator*(
      Eigen::MatrixBase<HPointDerived> const& p) const {
    const auto rp = *this * p.template head<M>();
    HomogeneousPointProduct<HPointDerived> r;
    r << rp, p(M);
    return r;
  }

  /// Group action on lines.
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), l.direction());
  }

  /// Group action on planes.
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    return Hyperplane(p.normal(), p.offset() - params().dot(p.normal()));
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this Cartesian's Scalar
  /// type.
  ///
  template <typename OtherDerived,
            typename = typename std::enable_if<
                std::is_same<Scalar, ReturnScalar<OtherDerived>>::value>::type>
  SOPHUS_FUNC CartesianBase<Derived, M>& operator*=(
      CartesianBase<OtherDerived, M> const& other) {
    *static_cast<Derived*>(this) = *this * other;
    return *this;
  }

  /// Mutator of params vector.
  ///
  SOPHUS_FUNC ParamsType& params() {
    return static_cast<Derived*>(this)->params();
  }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC ParamsType const& params() const {
    return static_cast<Derived const*>(this)->params();
  }
};

/// Cartesian using default storage; derived from CartesianBase.
template <class ScalarT, int M, int Options>
class Cartesian : public CartesianBase<Cartesian<ScalarT, M, Options>, M> {
  using Base = CartesianBase<Cartesian<ScalarT, M, Options>, M>;

 public:
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;
  static int constexpr kMatrixDim = Base::kMatrixDim;
  static int constexpr kPointDim = Base::kPointDim;

  using Scalar = ScalarT;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using ParamsMember = sophus::Vector<Scalar, M, Options>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC Cartesian& operator=(Cartesian const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes to zero vector.
  ///
  SOPHUS_FUNC Cartesian() { params_.setZero(); }

  /// Copy constructor
  ///
  SOPHUS_FUNC Cartesian(Cartesian const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class OtherDerived>
  SOPHUS_FUNC Cartesian(CartesianBase<OtherDerived, M> const& other)
      : params_(other.params()) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value,
                  "must be same Scalar type");
  }

  /// Accepts either M-vector or (M+1)x(M+1) matrices.
  ///
  template <class D>
  explicit SOPHUS_FUNC Cartesian(Eigen::MatrixBase<D> const& m) {
    static_assert(
        std::is_same<typename Eigen::MatrixBase<D>::Scalar, Scalar>::value, "");
    if (m.rows() == kDoF && m.cols() == 1) {
      // trick so this compiles
      params_ = m.template block<M, 1>(0, 0);
    } else if (m.rows() == kMatrixDim && m.cols() == kMatrixDim) {
      params_ = m.template block<M, 1>(0, M);
    } else {
      SOPHUS_ENSURE(false, "{} {}", m.rows(), m.cols());
    }
  }

  /// This provides unsafe read/write access to internal data.
  ///
  SOPHUS_FUNC Scalar* data() { return params_.data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC Scalar const* data() const { return params_.data(); }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static sophus::Matrix<Scalar, kNumParameters, kDoF>
  Dx_exp_x_at_0() {
    sophus::Matrix<Scalar, kNumParameters, kDoF> m;
    m.setIdentity();
    return m;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static sophus::Matrix<Scalar, kNumParameters, kDoF> Dx_exp_x(
      Tangent const&) {
    return Dx_exp_x_at_0();
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static sophus::Matrix<Scalar, kPointDim, kDoF> Dx_exp_x_times_point_at_0(
      Point const&) {
    sophus::Matrix<Scalar, kPointDim, kDoF> J;
    J.setIdentity();
    return J;
  }

  /// Returns derivative of exp(x).matrix() wrt. ``x_i at x=0``.
  ///
  SOPHUS_FUNC static Transformation Dxi_exp_x_matrix_at_0(int i) {
    return generator(i);
  }

  /// Mutator of params vector
  ///
  SOPHUS_FUNC ParamsMember& params() { return params_; }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC ParamsMember const& params() const { return params_; }

  /// Returns the ith infinitesimal generators of Cartesian(M).
  ///
  /// The infinitesimal generators for e.g. the 3-dimensional case:
  ///
  /// ```
  ///         |  0  0  0  1 |
  ///   G_0 = |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_1 = |  0  0  0  1 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_2 = |  0  0  0  0 |
  ///         |  0  0  0  1 |
  ///         |  0  0  0  0 |
  /// ```
  ///
  /// Precondition: ``i`` must be in [0, M-1].
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= M, "i should be in range [0,M-1].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// Group exponential
  ///
  /// For Euclidean vector space, just the identity. Or to be more precise
  /// it just constructs the (M+1xM+1) homogeneous matrix representation
  //  from the M-vector.
  ///
  SOPHUS_FUNC static Cartesian<Scalar, M> exp(Tangent const& a) {
    return Cartesian<Scalar, M>(a);
  }

  /// hat-operator
  ///
  /// Formally, the hat()-operator of Cartesian(M) is defined as
  ///
  ///   ``hat(.): R^M -> R^{M+1xM+1},  hat(a) = sum_i a_i * G_i``
  ///   (for i=0,...,M-1)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of Cartesian(M).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation Omega;
    Omega.setZero();
    Omega.col(M).template head<M>() = a.template head<M>();
    return Omega;
  }

  /// Lie bracket
  ///
  /// Always 0 for commutative groups.
  SOPHUS_FUNC static Tangent lieBracket(Tangent const&, Tangent const&) {
    return Tangent::Zero();
  }

  /// Draws uniform samples in the range [-1, 1] per coordinates.
  ///
  template <class UniformRandomBitGenerator>
  static Cartesian sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    Vector<Scalar, M> v;
    for (int i = 0; i < M; ++i) {
      v[i] = uniform(generator);
    }
    return Cartesian(v);
  }

  /// vee-operator
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& m) {
    return m.col(M).template head<M>();
  }

 protected:
  ParamsMember params_;
};

}  // namespace sophus

namespace Eigen {

/// Specialization of Eigen::Map for ``Cartesian``; derived from
/// CartesianBase.
///
/// Allows us to wrap Cartesian objects around POD array.
template <class ScalarT, int M, int Options>
class Map<sophus::Cartesian<ScalarT, M>, Options>
    : public sophus::CartesianBase<Map<sophus::Cartesian<ScalarT, M>, Options>,
                                   M> {
 public:
  using Base =
      sophus::CartesianBase<Map<sophus::Cartesian<ScalarT, M>, Options>, M>;
  using Scalar = ScalarT;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar* coeffs) : params_(coeffs) {}

  /// Mutator of params vector
  ///
  SOPHUS_FUNC Map<sophus::Vector<Scalar, M, Options>>& params() {
    return params_;
  }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC Map<sophus::Vector<Scalar, M, Options>> const& params() const {
    return params_;
  }

 protected:
  Map<sophus::Vector<Scalar, M>, Options> params_;
};

/// Specialization of Eigen::Map for ``Cartesian const``; derived from
/// CartesianBase.
///
/// Allows us to wrap Cartesian objects around POD array.
template <class ScalarT, int M, int Options>
class Map<sophus::Cartesian<ScalarT, M> const, Options>
    : public sophus::CartesianBase<
          Map<sophus::Cartesian<ScalarT, M> const, Options>, M> {
 public:
  using Base =
      sophus::CartesianBase<Map<sophus::Cartesian<ScalarT, M> const, Options>,
                            M>;
  using Scalar = ScalarT;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;

  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : params_(coeffs) {}

  /// Accessor of params vector
  ///
  SOPHUS_FUNC Map<sophus::Vector<Scalar, M> const, Options> const& params()
      const {
    return params_;
  }

 protected:
  Map<sophus::Vector<Scalar, M> const, Options> const params_;
};
}  // namespace Eigen
