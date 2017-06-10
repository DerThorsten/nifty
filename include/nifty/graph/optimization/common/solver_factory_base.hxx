#pragma once

namespace nifty {
namespace graph {
namespace optimization{
namespace common{




    template<class OBJECTIVE, class SOLVER_BASE>
    class SolverFactoryBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef SOLVER_BASE SolverBaseType;
        virtual ~SolverFactoryBase(){}
        virtual std::shared_ptr<SolverBaseType> createShared(const ObjectiveType & objective) = 0;
        virtual SolverBaseType * createRaw(const ObjectiveType & objective) = 0;
    };



} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

