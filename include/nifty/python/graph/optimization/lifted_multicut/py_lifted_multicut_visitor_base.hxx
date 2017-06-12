#pragma once

#include <cstddef>
#include <string>
#include <initializer_list>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{








template<class OBJECTIVE>
class PyLiftedMulticutVisitorBase : public LiftedMulticutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<Objective>::LiftedMulticutFactory;

    typedef OBJECTIVE Objective;
    typedef LiftedMulticutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef LiftedMulticutBase<Objective> LmcBase;
    typedef typename Objective::Graph Graph;



    void begin(LmcBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                       /* Return type */
            VisitorBaseType,                /* Parent class */
            begin,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    bool visit(LmcBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            bool,                       /* Return type */
            VisitorBaseType,                /* Parent class */
            visit,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    void end(LmcBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                    /* Return type */
            VisitorBaseType,             /* Parent class */
            end,                     /* Name of function */
            mcBase                   /* Argument(s) */
        );
    }

    void addLogNames(std::initializer_list<std::string> logNames) {
        PYBIND11_OVERLOAD(
            void,                    /* Return type */
            VisitorBaseType,             /* Parent class */
            addLogNames,                     /* Name of function */
            logNames                   /* Argument(s) */
        );
    }

    void setLogValue(const std::size_t logIndex, double logValue) {
        PYBIND11_OVERLOAD(
            void,                    /* Return type */
            VisitorBaseType,             /* Parent class */
            setLogValue,                     /* Name of function */
            logIndex,logValue        /* Argument(s) */
        );
    }

};

} // namespace lifted_multicut
} // namespace optimization
} // namespace graph
} // namespace nifty
