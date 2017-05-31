#pragma once
#ifndef NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_BASE_HXX
#define NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_BASE_HXX

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{







template<class OBJECTIVE>
class PyLiftedMulticutBase : public LiftedMulticutBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<Objective>::LiftedMulticutFactory;

    typedef OBJECTIVE Objective;
    typedef LiftedMulticutVisitorBase<OBJECTIVE> VisitorBase;
    typedef LiftedMulticutBase<Objective> McBase;
    typedef typename Objective::Graph Graph;
    typedef typename Graph:: template EdgeMap<uint8_t>  EdgeLabels;
    typedef typename Graph:: template NodeMap<uint64_t> NodeLabels;


    /* Trampoline (need one for each virtual function) */
    void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) {
        PYBIND11_OVERLOAD_PURE(
            void,                  /* Return type */
            McBase,                /* Parent class */
            optimize,              /* Name of function */
            nodeLabels,  visitor   /* Argument(s) */
        );
    }

    const NodeLabels & currentBestNodeLabels()  {
        PYBIND11_OVERLOAD_PURE(
            const NodeLabels &,                 /* Return type */
            McBase,                             /* Parent class */
            currentBestNodeLabels,              /* Name of function */
        );
    }

    double currentBestEnergy()  {
        PYBIND11_OVERLOAD_PURE(
            double,                  /* Return type */
            McBase,                  /* Parent class */
            currentBestEnergy,       /* Name of function */
        );
    }

    const Objective & objective() const {
        PYBIND11_OVERLOAD_PURE(
            const Objective & ,    /* Return type */
            McBase,                /* Parent class */
            objective              /* Name of function */
        );
    }

    std::string name() const{
        PYBIND11_OVERLOAD_PURE(
            std::string ,    /* Return type */
            McBase,          /* Parent class */
            name             /* Name of function */
        );
    }
};


} // namespace lifted_mutlcut
} // namespace optimization
} // namespace graph
} // namespace nifty

#endif /* NIFTY_PYTHON_GRAPH_LIFTED_MULTICUT_LIFTED_MULTICUT_BASE_HXX */
