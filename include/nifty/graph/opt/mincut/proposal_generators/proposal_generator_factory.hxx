
#pragma once


#include <memory>


#include "nifty/graph/opt/mincut/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace mincut{


    template<class PROPOSAL_GENERATOR>
    class ProposalGeneratorFactory : public ProposalGeneratorFactoryBase<typename PROPOSAL_GENERATOR::ObjectiveType>{
    public:
        typedef PROPOSAL_GENERATOR ProposalGeneratorType;       
        typedef typename ProposalGeneratorType::SettingsType SettingsType;
        typedef typename ProposalGeneratorType::ObjectiveType ObjectiveType;
        typedef ProposalGeneratorBase<ObjectiveType> ProposalGeneratorBaseType;

        ProposalGeneratorFactory(const SettingsType & settings = SettingsType())
        :   settings_(settings){

        }

        virtual ~ProposalGeneratorFactory(){}

        virtual std::shared_ptr<ProposalGeneratorBaseType> createShared(const ObjectiveType & objective,  const size_t numberOfThreads){
            return std::make_shared<ProposalGeneratorType>(objective, numberOfThreads, settings_);
        }
        virtual ProposalGeneratorBaseType *                create(   const ObjectiveType & objective,  const size_t numberOfThreads){
            return new ProposalGeneratorType(objective, numberOfThreads, settings_);
        }
    private:
        SettingsType settings_;
    };



}
}
}
}

