#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_FACTORY_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_FACTORY_HXX

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"

namespace nifty {
namespace graph {
namespace lifted_multicut{

    template<class OBJECTIVE>
    class LiftedMulticutBase;


    template<class OBJECTIVE>
    class LiftedMulticutFactoryBase{
    public:
        typedef OBJECTIVE Objective;
        typedef LiftedMulticutBase<Objective> LiftedMulticutBaseType;
        virtual ~LiftedMulticutFactoryBase(){}
        virtual std::shared_ptr<LiftedMulticutBaseType> createSharedPtr(const Objective & objective) = 0;
        virtual LiftedMulticutBaseType * createRawPtr(const Objective & objective) = 0;
    };


    template<class SOLVER>
    class LiftedMulticutFactory :
    public LiftedMulticutFactoryBase<typename SOLVER::Objective>{
    public:
        typedef typename SOLVER::Objective Objective;
        typedef LiftedMulticutBase<Objective> LiftedMulticutBaseType;
        typedef SOLVER Solver;
        typedef typename Solver::Settings Settings;
        LiftedMulticutFactory(const Settings & settings = Settings())
        :   LiftedMulticutFactoryBase<Objective>(),
            options_(settings){
        }
        virtual std::shared_ptr<LiftedMulticutBaseType> createSharedPtr(const Objective & objective){
            return std::make_shared<Solver>(objective, options_);
        }
        virtual LiftedMulticutBaseType * createRawPtr(const Objective & objective){
            LiftedMulticutBaseType *  p =  new Solver(objective, options_);
            return p;
        }
    private:
        Settings options_;
    };

} // namespace lifted_multicut
} // namespace graph
} // namespace nifty

#endif /* NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_FACTORY_HXX */
