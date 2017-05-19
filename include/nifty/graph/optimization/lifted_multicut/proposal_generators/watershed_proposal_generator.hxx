
#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_WATERSHED_PROPOSAL_GENERATOR_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_WATERSHED_PROPOSAL_GENERATOR_BASE_HXX

#include <vector>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/edge_weighted_watersheds.hxx"

namespace nifty{
namespace graph{
namespace lifted_multicut{


    /**
     * @brief      Watershed proposal generator for lifted_multicut::FusionMoveBased
     *
     * @tparam     OBJECTIVE  { description }
     */
    template<class OBJECTIVE>
    class WatershedProposalGenerator : public ProposalGeneratorBase<OBJECTIVE>{
    public:
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutBase<ObjectiveType> LiftedMulticutBaseType;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        typedef typename LiftedMulticutBaseType::NodeLabelsType NodeLabelsType;
    
        typedef typename GraphType:: template EdgeMap<float>  EdgeWeights;

        struct Settings{


            enum SeedingStrategie{
                SEED_FROM_LIFTED,
                SEED_FROM_LOCAL,
                SEED_FROM_BOTH
            };

            SeedingStrategie seedingStrategie{SEED_FROM_LIFTED};
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
            const auto & weights = objective_.weights();

            objective_.forEachGraphEdge([&](const uint64_t edge){
                const auto graphEdge = objective_.liftedGraphEdgeInGraph(edge);
                graphEdgeWeights_[graphEdge] = weights[edge];
            });

            if(settings_.seedingStrategie == Settings::SEED_FROM_LIFTED){
                objective_.forEachLiftedeEdge([&](const uint64_t edge){
                    if(weights[edge] < 0.0){
                        negativeEdges_.push_back(edge);
                    }
                });
                // if no negative lifted edges
                // use negative local edges
                if(negativeEdges_.size() == 0){
                    objective_.forEachGraphEdge([&](const uint64_t edge){
                        if(weights[edge] < 0.0){
                            negativeEdges_.push_back(edge);
                        }
                    });
                }
            }
            else if(settings_.seedingStrategie == Settings::SEED_FROM_LOCAL){
                objective_.forEachGraphEdge([&](const uint64_t edge){
                    if(weights[edge] < 0.0){
                        negativeEdges_.push_back(edge);
                    }
                });
                // if no negative local edges
                // use negative lifted edges
                if(negativeEdges_.size() == 0){
                    objective_.forEachLiftedeEdge([&](const uint64_t edge){
                        if(weights[edge] < 0.0){
                            negativeEdges_.push_back(edge);
                        }
                    });
                }
            }
            else if(settings_.seedingStrategie == Settings::SEED_FROM_BOTH){
                objective_.liftedGraph().forEachEdge([&](const uint64_t edge){
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
            const NodeLabelsType & currentBest,NodeLabelsType & proposal, 
            const size_t tid
        ){
            
            if(negativeEdges_.empty()){
                // do nothing
            }
            else{

                const auto & graph = objective_.graph();
                const auto & liftedGraph = objective_.liftedGraph();
                auto & gen = gens_[tid];

                EdgeWeights noisyEdgeWeights(graph);
                NodeLabelsType  seeds(graph, 0);

                auto nSeeds = settings_.numberOfSeeds <=1.0 ? 
                    size_t(float(graph.numberOfNodes())*settings_.numberOfSeeds+0.5f) :
                    size_t(settings_.numberOfSeeds + 0.5);

                nSeeds = std::max(size_t(1),nSeeds);
                nSeeds = std::min(size_t(negativeEdges_.size()-1), nSeeds);


                graph.forEachEdge([&](const uint64_t edge){
                    noisyEdgeWeights[edge] = graphEdgeWeights_[edge] + dist_(gen);
                });


                for(size_t i=0; i <  (nSeeds == 1 ? 1 : nSeeds/2); ++i){
                    const auto randIndex = intDist_(gen);
                    const auto edge  = negativeEdges_[randIndex];
                    const auto uv = liftedGraph.uv(edge);
          
                    seeds[uv.first] = (2*i)+1;
                    seeds[uv.second] = (2*i+1)+1;
                }

                edgeWeightedWatershedsSegmentation(graph, noisyEdgeWeights, seeds, proposal);

            }


        }
    private:
        const ObjectiveType & objective_;
        size_t numberOfThreads_;
        Settings settings_;
        std::vector<uint64_t> negativeEdges_;
        EdgeWeights graphEdgeWeights_;


        std::vector<std::mt19937> gens_;
        std::normal_distribution<> dist_;
        std::uniform_int_distribution<>  intDist_;

        
    }; 





}
}
}

#endif //NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_PROPOSAL_GENERATORS_WATERSHED_PROPOSAL_GENERATOR_BASE_HXX