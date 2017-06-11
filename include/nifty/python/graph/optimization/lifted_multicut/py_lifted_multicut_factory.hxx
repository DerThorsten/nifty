#pragma once

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_factory.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{






/*
    template<class MODEL>
    class LiftedMulticutFactoryBase{
    public:
        typedef MODEL Model;
        typedef LiftedMulticutBase<Model> LiftedMulticutBaseType;
        virtual ~LiftedMulticutFactoryBase(){}
        virtual std::shared_ptr<LiftedMulticutBaseType> create(const Model & model) = 0;
    };
*/





template<class OBJECTIVE>
class PyLiftedMulticutFactoryBase : public LiftedMulticutFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<Objective>::LiftedMulticutFactory;
    typedef OBJECTIVE Objective;
    typedef LiftedMulticutBase<Objective> LiftedMulticutBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<LiftedMulticutBaseType> createShared(const Objective & objective) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<LiftedMulticutBaseType>, /* Return type */
            LiftedMulticutFactoryBase<Objective>,    /* Parent class */
            createShared,                   /* Name of function */
            objective                          /* Argument(s) */
        );
    }
    LiftedMulticutBaseType * create(const Objective & objective) {
        PYBIND11_OVERLOAD_PURE(
            LiftedMulticutBaseType* ,                /* Return type */
            LiftedMulticutFactoryBase<Objective>,    /* Parent class */
            create,                            /* Name of function */
            objective                          /* Argument(s) */
        );
    }
};

} // namespace lifted_mutlicut
} // namespace optimization
} // namespace graph
} // namespace nifty

