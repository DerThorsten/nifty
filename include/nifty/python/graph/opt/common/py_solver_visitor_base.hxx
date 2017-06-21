#pragma once

#include <cstddef>
#include <string>
#include <initializer_list>

#include "nifty/graph/opt/multicut/multicut_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace multicut{






template<class OBJECTIVE>
class PyMulticutVisitorBase : public MulticutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MulticutFactory<ObjectiveType>::MulticutFactory;

    typedef OBJECTIVE ObjectiveType;
    typedef MulticutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef MulticutBase<ObjectiveType> McBase;
    typedef typename ObjectiveType::Graph Graph;



    void begin(McBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                       /* Return type */
            VisitorBaseType,                /* Parent class */
            begin,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    bool visit(McBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            bool,                       /* Return type */
            VisitorBaseType,                /* Parent class */
            visit,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    void end(McBase * mcBase) {
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

} // namespace multicut
} // namespace opt
} // namespace graph
} // namespace nifty
