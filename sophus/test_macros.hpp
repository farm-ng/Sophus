#pragma once

#include <iostream>
#include <sstream>

#include <sophus/types.hpp>

namespace sophus {
namespace details {

template <class Scalar, class Enable = void>
class Pretty {
 public:
  static std::string impl(Scalar s) {
    std::stringstream sstr;
    sstr << s;
    return sstr.str();
  }
};

template <class Ptr>
class Pretty<Ptr, enable_if_t<std::is_pointer<Ptr>::value>> {
 public:
  static std::string impl(Ptr ptr) {
    std::stringstream sstr;
    sstr << std::intptr_t(ptr);
    return sstr.str();
  }
};

template <class Scalar, int M, int kMatrixDim>
class Pretty<Eigen::Matrix<Scalar, M, kMatrixDim>, void> {
 public:
  static std::string impl(Matrix<Scalar, M, kMatrixDim> const& v) {
    std::stringstream sstr;
    sstr << "\n" << v << "\n";
    return sstr.str();
  }
};

template <class T>
std::string pretty(T const& v) {
  return Pretty<T>::impl(v);
}

template <class... Args>
void testFailed(bool& passed, char const* func, char const* file, int line,
                std::string const& msg) {
  std::fprintf(stderr,
               "Test failed in function '%s', "
               "file '%s', line %d.\n",
               func, file, line);
  std::cerr << msg << "\n\n";
  passed = false;
}

}  // namespace details

void processTestResult(bool passed) {
  if (!passed) {
    // LCOV_EXCL_START
    std::cerr << "failed!" << std::endl << std::endl;
    exit(-1);
    // LCOV_EXCL_STOP
  }
  std::cerr << "passed." << std::endl << std::endl;
}
}  // namespace sophus

#define SOPHUS_STRINGIFY(x) #x

/// Tests whether condition is true.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST(passed, condition, descr, ...)                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::string msg = SOPHUS_FMT_STR("condition ``{}`` is false\n",          \
                                       SOPHUS_STRINGIFY(condition));           \
      msg += SOPHUS_FMT_STR(descr, ##__VA_ARGS__);                             \
      sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_EQUAL(passed, left, right, descr, ...)                     \
  do {                                                                         \
    if (left != right) {                                                       \
      std::string msg = SOPHUS_FMT_STR(                                        \
          "{} (={}) is not equal to {} (={})\n", SOPHUS_STRINGIFY(left),       \
          sophus::details::pretty(left), SOPHUS_STRINGIFY(right),              \
          sophus::details::pretty(right));                                     \
      msg += SOPHUS_FMT_STR(descr, ##__VA_ARGS__);                             \
      sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_NEQ(passed, left, right, descr, ...)                       \
  do {                                                                         \
    if (left == right) {                                                       \
      std::string msg = SOPHUS_FMT_STR(                                        \
          "{} (={}) should not be equal to {} (={})\n",                        \
          SOPHUS_STRINGIFY(left), sophus::details::pretty(left),               \
          SOPHUS_STRINGIFY(right), sophus::details::pretty(right));            \
      msg += SOPHUS_FMT_STR(descr, ##__VA_ARGS__);                             \
      sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is approximately equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_APPROX(passed, left, right, thr, descr, ...)               \
  do {                                                                         \
    auto nrm = sophus::maxMetric((left), (right));                             \
    if (!(nrm < (thr))) {                                                      \
      std::string msg = SOPHUS_FMT_STR(                                        \
          "{} (={}) is not approx {} (={}); {} is {}; nrm is {}\n",            \
          SOPHUS_STRINGIFY(left), sophus::details::pretty(left),               \
          SOPHUS_STRINGIFY(right), sophus::details::pretty(right),             \
          SOPHUS_STRINGIFY(thr), sophus::details::pretty(thr), nrm);           \
      msg += SOPHUS_FMT_STR(descr, ##__VA_ARGS__);                             \
      sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is NOT approximately equal to right given a
/// threshold. The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_NOT_APPROX(passed, left, right, thr, descr, ...)           \
  do {                                                                         \
    auto nrm = sophus::maxMetric((left), (right));                             \
    if (nrm < (thr)) {                                                         \
      std::string msg = SOPHUS_FMT_STR(                                        \
          "{} (={}) is approx {} (={}), but it should not!\n {} is {}; nrm "   \
          "is {}\n",                                                           \
          SOPHUS_STRINGIFY(left), sophus::details::pretty(left),               \
          SOPHUS_STRINGIFY(right), sophus::details::pretty(right),             \
          SOPHUS_STRINGIFY(thr), sophus::details::pretty(thr), nrm);           \
      msg += SOPHUS_FMT_STR(descr, ##__VA_ARGS__);                             \
      sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)
