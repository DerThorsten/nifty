
#pragma once



#include <memory>


#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_base.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace common{



    template<class PROPOSAL_GENERATOR, class NODE_LABELS>
    class ProposalGeneratorFactory : public ProposalGeneratorFactoryBase<NODE_LABELS>{
    public:
        typedef PROPOSAL_GENERATOR                              ProposalGeneratorType;       
        typedef typename ProposalGeneratorType::Settings        Settings;
        typedef typename ProposalGeneratorType::ObjectiveType   ObjectiveType;
        typedef ProposalGeneratorBase<NODE_LABELS>              ProposalGeneratorBaseType;

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
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_PROPOSAL_GENERATOR_FACTORY_HXX