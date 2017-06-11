#pragma once

#include "nifty/graph/optimization/common/solver_factory_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace common{







template<class OBJECTIVE, class SOLVER_BASE>
class PyMulticutFactoryBase : public MulticutFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MulticutFactory<ObjectiveType>::MulticutFactory;
    typedef OBJECTIVE ObjectiveType;
    typedef SOLVER_BASE SolverBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<SolverBaseType> createShared(const ObjectiveType & objective) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<SolverBaseType>, /* Return type */
            MulticutFactoryBase<ObjectiveType>,    /* Parent class */
            createShared,                   /* Name of function */
            objective                          /* Argument(s) */
        );
    }
    SolverBaseType * create(const ObjectiveType & objective) {
        PYBIND11_OVERLOAD_PURE(
            SolverBaseType* ,                /* Return type */
            MulticutFactoryBase<ObjectiveType>,    /* Parent class */
            create,                            /* Name of function */
            objective                          /* Argument(s) */
        );
    }
};

} // namespace common
} // namespace optimization
} // namespace graph
} // namespace nifty

