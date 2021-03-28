#include "ceres/ceres.h"
#include "glog/logging.h"
#define AUTODIFF 0
#define NUMEDIFF 0
#define ANALDIFF 1

using namespace std;

#if AUTODIFF
struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
#endif

#if NUMEDIFF
struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
#endif

#if ANALDIFF
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  virtual ~QuadraticCostFunction() {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    double x = parameters[0][0];
    residuals[0] = 10 - x;
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};
#endif

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double x = double(int(*argv[1] - '0'));
  const double initial_x = x;

  ceres::Problem problem;

#if AUTODIFF
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);
#endif

#if NUMEDIFF
  ceres::CostFunction* cost_function =
      new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL,
                                         1, 1>(new NumericDiffCostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);
#endif

#if ANALDIFF
  ceres::CostFunction* cost_function = new QuadraticCostFunction;
  problem.AddResidualBlock(cost_function, NULL, &x);
#endif

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x: " << initial_x << "->" << x << "\n";

  return 0;
}