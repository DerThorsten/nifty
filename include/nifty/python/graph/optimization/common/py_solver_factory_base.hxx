#pragma once

#include <memory>

#include "nifty/graph/optimization/common/solver_factory_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace common{







template<class SOLVER_BASE>
class PySolverFactoryBase : public SolverFactoryBase<SOLVER_BASE> {
public:
    /* Inherit the constructors */
    // using SolverFactory<ObjectiveType>::SolverFactory;
    
    typedef SolverFactoryBase<SOLVER_BASE> BaseType;
    typedef SOLVER_BASE SolverBaseType;
    typedef typename SolverBaseType::ObjectiveType ObjectiveType;
    
    /* Trampoline (need one for each virtual function) */



    std::shared_ptr<SolverBaseType> createShared(const ObjectiveType & objective) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<SolverBaseType>, /* Return type */
            BaseType,                        /* Parent class */
            createShared,                    /* Name of function */
            objective                        /* Argument(s) */
        );
    }

    SolverBaseType * create(const ObjectiveType & objective) {
        PYBIND11_OVERLOAD_PURE(
            SolverBaseType* ,                /* Return type */
            BaseType,                        /* Parent class */
            create,                          /* Name of function */
            objective                        /* Argument(s) */
        );
    }

};

} // namespace common
} // namespace optimization
} // namespace graph
} // namespace nifty

