
#pragma once

#include <vector>

#include "nifty/graph/opt/mincut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace mincut{


    /**
     * @brief      Watershed proposal generator for mincut::FusionMoveBased
     *
     * @tparam     OBJECTIVE  { description }
     */
    template<class OBJECTIVE>
    class RandomProposalGenerator : public ProposalGeneratorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef MincutBase<ObjectiveType> MincutBaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        typedef typename MincutBaseType::NodeLabels NodeLabels;
    
        typedef typename GraphType:: template EdgeMap<float>  EdgeWeights;

        struct SettingsType{


            enum SeedingStrategie{
                SEED_FROM_LIFTED,
                SEED_FROM_LOCAL,
                SEED_FROM_BOTH
            };

            SeedingStrategie seedingStrategie{SEED_FROM_LIFTED};
            double sigma{1.0};
            double numberOfSeeds{0.1};
        };

        RandomProposalGenerator(
            const ObjectiveType & objective, 
            const size_t numberOfThreads,
            const SettingsType & settings  = SettingsType()
        )
        :   objective_(objective),
            numberOfThreads_(numberOfThreads),
            settings_(settings),
            negativeEdges_(),
            graphEdgeWeights_(objective.graph()),
            gens_(numberOfThreads_),
            dist_(0.0, settings.sigma),
            intDist_()
        {
            // use thread index as seed
            for(auto i=0; i<numberOfThreads_; ++i)
                gens_[i] = std::mt19937(i);

            this->reset();
        }

        void reset(){

        }

        virtual ~RandomProposalGenerator(){}

        virtual void generateProposal(
            const NodeLabels & currentBest,NodeLabels & proposal, 
            const size_t tid
        ){
            
        }
    private:
        const ObjectiveType & objective_;
        size_t numberOfThreads_;
        SettingsType settings_;
        std::vector<std::mt19937> gens_;    
    }; 





}
}
}
}

