#pragma once

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{

    template<class OBJECTIVE>
    class LiftedMulticutBase;


    template<class OBJECTIVE>
    class LiftedMulticutFactoryBase{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> LiftedMulticutBaseType;
        virtual ~LiftedMulticutFactoryBase(){}
        virtual std::shared_ptr<LiftedMulticutBaseType> createShared(const ObjectiveType & objective) = 0;
        virtual LiftedMulticutBaseType * create(const ObjectiveType & objective) = 0;
    };


    template<class SOLVER>
    class LiftedMulticutFactory :
    public LiftedMulticutFactoryBase<typename SOLVER::ObjectiveType>{
    public:
        typedef typename SOLVER::ObjectiveType ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> LiftedMulticutBaseType;
        typedef SOLVER Solver;
        typedef typename Solver::Settings Settings;
        LiftedMulticutFactory(const Settings & settings = Settings())
        :   LiftedMulticutFactoryBase<ObjectiveType>(),
            options_(settings){
        }
        virtual std::shared_ptr<LiftedMulticutBaseType> createShared(const ObjectiveType & objective){
            return std::make_shared<Solver>(objective, options_);
        }
        virtual LiftedMulticutBaseType * create(const ObjectiveType & objective){
            LiftedMulticutBaseType *  p =  new Solver(objective, options_);
            return p;
        }
    private:
        Settings options_;
    };

} // namespace lifted_multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty
