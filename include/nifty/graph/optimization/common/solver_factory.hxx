#pragma once

#include "nifty/graph/optimization/common/solver_factory_base.hxx"


namespace nifty {
namespace graph {
namespace optimization{
namespace common{





    template<class SOLVER>
    class SolverFactory :
    public SolverFactoryBase<
        typename SOLVER::ObjectiveType,
        typename SOLVER::BaseType
    >{
    public: 
        typedef SolverFactoryBase< SOLVER::ObjectiveType, SOLVER::BaseType> BaseType;
        typedef SOLVER                                                      SolverType;
        typedef typename SolverType::ObjectiveType                          ObjectiveType;
        typedef typename SolverType::BaseType                               SolverBaseType;
        typedef typename SolverType::SettingsType                           SettingsType; 

        SolverFactory(const SettingsType & settings = SettingsType())
        :   BaseType<ObjectiveType>(),
            options_(settings){
        }
        virtual std::shared_ptr<SolverBaseType> createShared(const ObjectiveType & objective){
            return std::make_shared<Solver>(objective, options_);
        }
        virtual SolverBaseType * create(const ObjectiveType & objective){
            SolverBaseType *  p =  new Solver(objective, options_);
            return p;
        }
    private:
        SettingsType options_;
    };


} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

