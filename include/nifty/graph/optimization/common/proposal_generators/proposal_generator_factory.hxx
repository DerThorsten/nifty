
#pragma once



#include <memory>


#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace common{



    template<class PROPOSAL_GENERATOR>
    class ProposalGeneratorFactory : 
        public ProposalGeneratorFactoryBase<typename PROPOSAL_GENERATOR::ObjectiveType>{
    public:
        typedef PROPOSAL_GENERATOR                              ProposalGeneratorType;       
        typedef typename ProposalGeneratorType::Settings        Settings;
        typedef typename ProposalGeneratorType::ObjectiveType   ObjectiveType;
        typedef ProposalGeneratorBase<ObjectiveType>            ProposalGeneratorBaseType;

        ProposalGeneratorFactory(const Settings & settings = Settings())
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
        Settings settings_;
    };


} // namespace nifty::graph::optimization::common
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

