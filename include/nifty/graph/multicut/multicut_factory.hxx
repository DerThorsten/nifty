#pragma once
#ifndef NIFTY_GRAPH_MULTICUT_MULTICUT_FACTORY_HXX
#define NIFTY_GRAPH_MULTICUT_MULTICUT_FACTORY_HXX

#include "nifty/graph/multicut/multicut_base.hxx"

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class MulticutBase;


    template<class OBJECTIVE>
    class MulticutFactoryBase{
    public:
        typedef OBJECTIVE Objective;
        typedef MulticutBase<Objective> MulticutBaseType;
        virtual ~MulticutFactoryBase(){}
        virtual std::shared_ptr<MulticutBaseType> createSharedPtr(const Objective & objective) = 0;
        virtual MulticutBaseType * createRawPtr(const Objective & objective) = 0;
    };


    template<class SOLVER>
    class MulticutFactory :
    public MulticutFactoryBase<typename SOLVER::Objective>{
    public:
        typedef typename SOLVER::Objective Objective;
        typedef MulticutBase<Objective> MulticutBaseType;
        typedef SOLVER Solver;
        typedef typename Solver::Settings Settings;
        MulticutFactory(const Settings & settings = Settings())
        :   MulticutFactoryBase<Objective>(),
            options_(settings){
        }
        virtual std::shared_ptr<MulticutBaseType> createSharedPtr(const Objective & objective){
            return std::make_shared<Solver>(objective, options_);
        }
        virtual MulticutBaseType * createRawPtr(const Objective & objective){
            MulticutBaseType *  p =  new Solver(objective, options_);
            return p;
        }
    private:
        Settings options_;
    };

} // namespace graph
} // namespace nifty

#endif /* NIFTY_GRAPH_MULTICUT_MULTICUT_FACTORY_HXX */
