#pragma once

#include <cstddef>
#include <string>
#include <initializer_list>

#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace ho_multicut{






template<class OBJECTIVE>
class PyHoMulticutVisitorBase : public HoMulticutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using HoMulticutFactory<ObjectiveType>::HoMulticutFactory;

    typedef OBJECTIVE ObjectiveType;
    typedef HoMulticutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef HoMulticutBase<ObjectiveType> McBase;
    typedef typename ObjectiveType::GraphType GraphType;



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

} // namespace ho_multicut
} // namespace opt
} // namespace graph
} // namespace nifty
