#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_PY_MULTICUT_FACTORY_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_PY_MULTICUT_FACTORY_HXX

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_factory.hxx"

namespace nifty {
namespace graph {






/*
    template<class MODEL>
    class MulticutFactoryBase{
    public:
        typedef MODEL Model;
        typedef MulticutBase<Model> MulticutBaseType;
        virtual ~MulticutFactoryBase(){}
        virtual std::shared_ptr<MulticutBaseType> create(const Model & model) = 0;
    };
*/





template<class OBJECTIVE>
class PyMulticutFactoryBase : public MulticutFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MulticutFactory<Objective>::MulticutFactory;
    typedef OBJECTIVE Objective;
    typedef MulticutBase<Objective> MulticutBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<MulticutBaseType> createSharedPtr(const Objective & objective) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<MulticutBaseType>, /* Return type */
            MulticutFactoryBase<Objective>,    /* Parent class */
            createSharedPtr,                   /* Name of function */
            objective                          /* Argument(s) */
        );
    }
    MulticutBaseType * createRawPtr(const Objective & objective) {
        PYBIND11_OVERLOAD_PURE(
            MulticutBaseType* ,                /* Return type */
            MulticutFactoryBase<Objective>,    /* Parent class */
            createRawPtr,                            /* Name of function */
            objective                          /* Argument(s) */
        );
    }
};


} // namespace graph
} // namespace nifty

#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_PY_MULTICUT_FACTORY_HXX */
