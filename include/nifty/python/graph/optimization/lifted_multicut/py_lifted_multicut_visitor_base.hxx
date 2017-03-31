#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_MULTICUT_VISITOR_BASE_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_MULTICUT_VISITOR_BASE_HXX

#include <string>
#include <initializer_list>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace lifted_multicut{








template<class OBJECTIVE>
class PyLiftedMulticutVisitorBase : public LiftedMulticutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<Objective>::LiftedMulticutFactory;

    typedef OBJECTIVE Objective;
    typedef LiftedMulticutVisitorBase<OBJECTIVE> VisitorBase;
    typedef LiftedMulticutBase<Objective> LmcBase;
    typedef typename Objective::Graph Graph;



    void begin(LmcBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                       /* Return type */
            VisitorBase,                /* Parent class */
            begin,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    bool visit(LmcBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            bool,                       /* Return type */
            VisitorBase,                /* Parent class */
            visit,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    void end(LmcBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                    /* Return type */
            VisitorBase,             /* Parent class */
            end,                     /* Name of function */
            mcBase                   /* Argument(s) */
        );
    }

    void addLogNames(std::initializer_list<std::string> logNames) {
        PYBIND11_OVERLOAD(
            void,                    /* Return type */
            VisitorBase,             /* Parent class */
            addLogNames,                     /* Name of function */
            logNames                   /* Argument(s) */
        );
    }

    void setLogValue(const size_t logIndex, double logValue) {
        PYBIND11_OVERLOAD(
            void,                    /* Return type */
            VisitorBase,             /* Parent class */
            setLogValue,                     /* Name of function */
            logIndex,logValue        /* Argument(s) */
        );
    }

};

} // namespace lifted_multicut
} // namespace graph
} // namespace nifty

#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_MULTICUT_VISITOR_BASE_HXX */
