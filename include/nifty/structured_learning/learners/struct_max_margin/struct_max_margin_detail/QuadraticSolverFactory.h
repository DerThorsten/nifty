#ifndef OPENGM_LEARNING_SOLVER_QUADRATIC_SOLVER_FACTORY_H__
#define OPENGM_LEARNING_SOLVER_QUADRATIC_SOLVER_FACTORY_H__

#include "QuadraticSolverBackend.h"
#ifdef WITH_GUROBI
#include "GurobiBackend.h"
#elif defined(WITH_CPLEX)
#include "CplexBackend.h"
#endif



namespace nifty {
namespace structured_learning {
namespace solver {

class QuadraticSolverFactory {

public:

	static QuadraticSolverBackend* Create() {

#ifdef WITH_GUROBI
	return new GurobiBackend();
#elif defined(WITH_CPLEX)
        return new CplexBackend();
#endif

      throw std::runtime_error("No quadratic solver available.");
	}
};

}}} // namespace opengm::learning::solver

#endif // OPENGM_LEARNING_SOLVER_QUADRATIC_SOLVER_FACTORY_H__

