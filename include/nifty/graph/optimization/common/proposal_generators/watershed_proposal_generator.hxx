
#pragma once

#include <string>
#include <random>
#include <vector>

#include "nifty/graph/optimization/common/proposal_generators/proposal_generator_base.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"


namespace nifty{
namespace graph{
namespace optimization{
namespace common{


    /**
     * @brief      Watershed proposal generator for lifted_multicut::FusionMoveBased
     *
     * @tparam     OBJECTIVE  { description }
     */
    template<class OBJECTIVE>
    class WatershedProposalGenerator : 
        public ProposalGeneratorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename GraphType:: template NodeMap<uint64_t>  ProposalType;
        typedef typename GraphType:: template EdgeMap<float>       EdgeWeights;

        struct Settings{


            enum SeedingStrategie{
                SEED_FROM_NEGATIVE,
                SEED_FROM_ALL
            };

            SeedingStrategie seedingStrategie{SEED_FROM_NEGATIVE};
            double sigma{1.0};
            double numberOfSeeds{0.1};
        };

        WatershedProposalGenerator(
            const ObjectiveType & objective, 
            const size_t numberOfThreads,
            const Settings & settings  = Settings()
        )
        :   objective_(objective),
            numberOfThreads_(numberOfThreads),
            settings_(settings),
            negativeEdges_(),
            seeds_(objective.graph()),
            noisyEdgeWeights_(objective.graph()),
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
            const auto & weights = objective_.weights();

            
            if(settings_.seedingStrategie == Settings::SEED_FROM_NEGATIVE){
                objective_.graph().forEachEdge([&](const uint64_t edge){
                    if(weights[edge] < 0.0){
                        negativeEdges_.push_back(edge);
                    }
                });
            }
            
            if(!negativeEdges_.empty())
                intDist_ = std::uniform_int_distribution<> (0, negativeEdges_.size()-1);
            else{
                // fallback to not crash, but meaningless since there are no negative edges
                intDist_ = std::uniform_int_distribution<> (0, 1);
            }
        }

        virtual ~WatershedProposalGenerator(){}

        virtual void generateProposal(
            const ProposalType & currentBest, 
            ProposalType & proposal, 
            const size_t tid
        ){
            
            if(negativeEdges_.empty()){
                // do nothing
            }
            else{

                const auto & graph = objective_.graph();

                auto & gen = gens_[tid];

                for(const auto node: graph.nodes()){
                    seeds_[node] = 0;
                }

                auto nSeeds = settings_.numberOfSeeds <=1.0 ? 
                    size_t(float(graph.numberOfNodes())*settings_.numberOfSeeds+0.5f) :
                    size_t(settings_.numberOfSeeds + 0.5);

                nSeeds = std::max(size_t(1),nSeeds);
                nSeeds = std::min(size_t(negativeEdges_.size()-1), nSeeds);


                const auto & weights = objective_.weights();
                graph.forEachEdge([&](const uint64_t edge){
                    noisyEdgeWeights_[edge] = weights[edge] + dist_(gen);
                });


                for(size_t i=0; i <  (nSeeds == 1 ? 1 : nSeeds/2); ++i){
                    const auto randIndex = intDist_(gen);
                    const auto edge  = negativeEdges_[randIndex];
                    const auto uv = graph.uv(edge);
          
                    seeds_[uv.first] = (2*i)+1;
                    seeds_[uv.second] = (2*i+1)+1;
                }

                edgeWeightedWatershedsSegmentation(graph, noisyEdgeWeights_, seeds_, proposal);

            }


        }
    private:
        const ObjectiveType & objective_;
        size_t numberOfThreads_;
        Settings settings_;
        std::vector<uint64_t> negativeEdges_;
        EdgeWeights noisyEdgeWeights_;
        ProposalType  seeds_;

        std::vector<std::mt19937> gens_;
        std::normal_distribution<> dist_;
        std::uniform_int_distribution<>  intDist_;

        
    }; 




}
}
}
}

