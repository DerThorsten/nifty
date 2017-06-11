#pragma once

#include "nifty/graph/optimization/mincut/mincut_base.hxx"
#include "nifty/graph/optimization/mincut/mincut_factory.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace mincut{




/*
    template<class MODEL>
    class MincutFactoryBase{
    public:
        typedef MODEL Model;
        typedef MincutBase<Model> MincutBaseType;
        virtual ~MincutFactoryBase(){}
        virtual std::shared_ptr<MincutBaseType> create(const Model & model) = 0;
    };
*/





template<class OBJECTIVE>
class PyMincutFactoryBase : public MincutFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MincutFactory<Objective>::MincutFactory;
    typedef OBJECTIVE Objective;
    typedef MincutBase<Objective> MincutBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<MincutBaseType> createShared(const Objective & objective) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<MincutBaseType>, /* Return type */
            MincutFactoryBase<Objective>,    /* Parent class */
            createShared,                   /* Name of function */
            objective                          /* Argument(s) */
        );
    }
    MincutBaseType * create(const Objective & objective) {
        PYBIND11_OVERLOAD_PURE(
            MincutBaseType* ,                /* Return type */
            MincutFactoryBase<Objective>,    /* Parent class */
            create,                            /* Name of function */
            objective                          /* Argument(s) */
        );
    }
};

} // namespace mincut
} // namespace optimization
} // namespace graph
} // namespace nifty

