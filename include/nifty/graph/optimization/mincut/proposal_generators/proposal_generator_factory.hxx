
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_HXX


#include <memory>


#include "nifty/graph/optimization/mincut/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace mincut{


    template<class PROPOSAL_GENERATOR>
    class ProposalGeneratorFactory : public ProposalGeneratorFactoryBase<typename PROPOSAL_GENERATOR::ObjectiveType>{
    public:
        typedef PROPOSAL_GENERATOR ProposalGeneratorType;       
        typedef typename ProposalGeneratorType::Settings Settings;
        typedef typename ProposalGeneratorType::ObjectiveType ObjectiveType;
        typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;

        ProposalGeneratorFactory(const Settings & settings = Settings())
        :   settings_(settings){

        }

        virtual ~ProposalGeneratorFactory(){}

        virtual std::shared_ptr<ProposalGeneratorBaseType> createSharedPtr(const ObjectiveType & objective,  const size_t numberOfThreads){
            return std::make_shared<ProposalGeneratorType>(objective, numberOfThreads, settings_);
        }
        virtual ProposalGeneratorBaseType *                createRawPtr(   const ObjectiveType & objective,  const size_t numberOfThreads){
            return new ProposalGeneratorType(objective, numberOfThreads, settings_);
        }
    private:
        Settings settings_;
    };



}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_HXX