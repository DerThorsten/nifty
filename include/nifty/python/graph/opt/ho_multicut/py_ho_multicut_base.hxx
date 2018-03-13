#pragma once

#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace ho_multicut{






template<class OBJECTIVE>
class PyHoMulticutBase : public HoMulticutBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using HoMulticutFactory<ObjectiveType>::HoMulticutFactory;

    typedef OBJECTIVE ObjectiveType;
    typedef HoMulticutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef HoMulticutBase<ObjectiveType> McBase;
    typedef typename ObjectiveType::GraphType GraphType;
    typedef typename GraphType:: template EdgeMap<uint8_t>  EdgeLabels;
    typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;


    /* Trampoline (need one for each virtual function) */
    void optimize(NodeLabelsType & nodeLabels, VisitorBaseType * visitor) {
        PYBIND11_OVERLOAD_PURE(
            void,                  /* Return type */
            McBase,                /* Parent class */
            optimize,              /* Name of function */
            nodeLabels,  visitor   /* Argument(s) */
        );
    }

    const NodeLabelsType & currentBestNodeLabels()  {
        PYBIND11_OVERLOAD_PURE(
            const NodeLabelsType &,                 /* Return type */
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

    const ObjectiveType & objective() const {
        PYBIND11_OVERLOAD_PURE(
            const ObjectiveType & ,    /* Return type */
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

} // namespace ho_multicut
} // namespace opt
} // namespace graph
} // namespace nifty

