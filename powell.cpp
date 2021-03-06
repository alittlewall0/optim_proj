#include <math.h>

#include <vector>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

struct F1 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x2, T* residual) const {
    residual[0] = x1[0] + 10.0 * x2[0];
    return true;
  }
};

struct F2 {
  template <typename T>
  bool operator()(const T* const x3, const T* const x4, T* residual) const {
    residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
    return true;
  }
};

struct F3 {
  template <typename T>
  bool operator()(const T* const x2, const T* const x3, T* residual) const {
    residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
    return true;
  }
};

struct F4 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x4, T* residual) const {
    residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

DEFINE_string(
    minimizer, "trust_region",
    "Minimizer type to use,choices are:line_search or trust_region");
DEFINE_double(x1, 3.0, "x1");
DEFINE_double(x2, 1.0, "x2");
DEFINE_double(x3, 7.0, "x3");
DEFINE_double(x4, 2.0, "x4");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  double x1 = FLAGS_x1;
  double x2 = FLAGS_x2;
  double x3 = FLAGS_x3;
  double x4 = FLAGS_x4;

  ceres::Problem problem;
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1),
                           NULL, &x1, &x2);
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                           NULL, &x3, &x4);
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                           NULL, &x2, &x3);
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
                           NULL, &x1, &x4);

  ceres::Solver::Options options;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer,
                                              &options.minimizer_type))
      << "Invalid minimizer:" << FLAGS_minimizer
      << ", valid are line search or trust region";

  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  std::cout << "Initial x1= " << x1 << ", x2= " << x2 << ", x3= " << x3
            << ", x4= " << x4 << "\n";

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "Final x1 x2 x3 x4 :" << x1 << " " << x2 << " " << x3 << " "
            << x4 << "\n";

  return 0;
}
