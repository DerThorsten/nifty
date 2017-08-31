#pragma once

#include "nifty/graph/opt/common/solver_factory_base.hxx"


namespace nifty {
namespace graph {
namespace opt{
namespace common{





    template<class SOLVER>
    class SolverFactory :
    public SolverFactoryBase<
        typename SOLVER::BaseType
    >{
    public: 
        typedef SolverFactoryBase<typename SOLVER::BaseType>    BaseType;
        typedef SOLVER                                          SolverType;
        typedef typename SolverType::ObjectiveType              ObjectiveType;
        typedef typename SolverType::BaseType                   SolverBaseType;
        typedef typename SolverType::SettingsType               SettingsType; 

        SolverFactory(const SettingsType & settings = SettingsType())
        :   BaseType(),
            options_(settings){
        }
        virtual std::shared_ptr<SolverBaseType> createShared(const ObjectiveType & objective){
            return std::make_shared<SolverType>(objective, options_);
        }
        virtual SolverBaseType * create(const ObjectiveType & objective){
            SolverBaseType *  p =  new SolverType(objective, options_);
            return p;
        }
    private:
        SettingsType options_;
    };


} // namespace nifty::graph::opt::common
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

