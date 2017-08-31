#pragma once

#include <cstddef>
#include <string>
#include <initializer_list>

#include "nifty/graph/opt/mincut/mincut_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace mincut{






template<class OBJECTIVE>
class PyMincutVisitorBase : public MincutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MincutFactory<Objective>::MincutFactory;

    typedef OBJECTIVE Objective;
    typedef MincutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef MincutBase<Objective> McBase;
    typedef typename Objective::Graph Graph;



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

} // namespace mincut
} // namespace opt
} // namespace graph
} // namespace nifty
