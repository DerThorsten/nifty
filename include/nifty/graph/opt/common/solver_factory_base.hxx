#pragma once

namespace nifty {
namespace graph {
namespace opt{
namespace common{




    template<class SOLVER_BASE>
    class SolverFactoryBase{
    public:
        
        typedef SOLVER_BASE SolverBaseType;
        typedef typename SolverBaseType::ObjectiveType ObjectiveType;
        virtual ~SolverFactoryBase(){}
        virtual std::shared_ptr<SolverBaseType> createShared(const ObjectiveType & objective) = 0;
        virtual SolverBaseType * create(const ObjectiveType & objective) = 0;
    };



} // namespace nifty::graph::opt::common
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

