#pragma once

#include <cstddef>
#include <string>
#include <initializer_list>

#include "nifty/graph/opt/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace lifted_multicut{








template<class OBJECTIVE>
class PyLiftedMulticutVisitorBase : public LiftedMulticutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<ObjectiveType>::LiftedMulticutFactory;

    typedef OBJECTIVE ObjectiveType;
    typedef LiftedMulticutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef LiftedMulticutBase<ObjectiveType> LmcBase;
    typedef typename ObjectiveType::GraphType GraphType;



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
} // namespace opt
} // namespace graph
} // namespace nifty
