#pragma once

#include "nifty/graph/optimization/multicut/multicut_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace multicut{

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
        typedef typename SOLVER::ObjectiveType ObjectiveType;
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
} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

