#pragma once

#include "nifty/graph/opt/mincut/mincut_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace mincut{






template<class OBJECTIVE>
class PyMincutBase : public MincutBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MincutFactory<Objective>::MincutFactory;

    typedef OBJECTIVE Objective;
    typedef MincutVisitorBase<OBJECTIVE> VisitorBaseType;
    typedef MincutBase<Objective> McBase;
    typedef typename Objective::Graph Graph;
    typedef typename McBase::NodeLabelsType NodeLabelsType;


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

} // namespace mincut
} // namespace opt
} // namespace graph
} // namespace nifty

