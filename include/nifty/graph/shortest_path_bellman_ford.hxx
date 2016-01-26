#pragma once
#ifndef NIFTY_GRAPH_SHORTEST_PATH_HXX
#define NIFTY_GRAPH_SHORTEST_PATH_HXX

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/queue/changeable_priority_queue.hxx"

namespace nifty{
namespace graph{


    struct CycleWithNegativeWeightsDetectedError : public std::runtime_error{
        CycleWithNegativeWeightsDetectedError():
        std::runtime_error("Nifty Error: CycleWithNegativeWeightsDetectedError"){
            
        }
    };

    template<class GRAPH, class WEIGHT_TYPE>
    class ShortestPathBellmanFord{

    public:
        typedef GRAPH Graph;
        typedef WEIGHT_TYPE WeightType;

        typedef typename Graph:: template NodeMap<int64_t>     PredecessorsMap;
        typedef typename Graph:: template NodeMap<WeightType>  DistanceMap;

    public:
        ShortestPathBellmanFord(const Graph & g)
        :   g_(g),
            predMap_(g),
            distMap_(g){
        }

        // run single source single target
        // no  callback no mask exposed
        template<class ArcWeights>
        void runSingleSourceSingleTarget(
            ArcWeights arcWeights,
            const int64_t source,
            const int64_t target = -1
        ){
            // subgraph mask
            DefaultSubgraphMask<Graph> subgraphMask;
            // visitor
            auto visitor = [&]
            (   
                int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return topNode != target;
            };

            this->initializeMaps(&source, &source +1);
            runImpl(arcWeights, subgraphMask, visitor);
        }

        // run single source  ALL targets
        // no  callback no mask exposed
        template<class ArcWeights>
        void runSingleSource(
            ArcWeights arcWeights,
            const int64_t source
        ){

            // subgraph mask
            DefaultSubgraphMask<Graph> subgraphMask;
            this->initializeMaps(&source, &source +1);
            // visitor
            auto visitor = [](   int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return true;
            };
            runImpl(arcWeights, subgraphMask, visitor);
        }

        template<class ArcWeights, class SOURCE_ITER, class SUBGRAPH_MASK, class VISITOR>
        void run(
            ArcWeights arcWeights,
            SOURCE_ITER sourceBegin, 
            SOURCE_ITER sourceEnd,
            const SUBGRAPH_MASK &  subgraphMask,
            VISITOR && visitor
        ){
            this->initializeMaps(sourceBegin, sourceEnd);
            this->runImpl(arcWeights,subgraphMask,visitor);
        }

        const DistanceMap & distances()const{
            return distMap_;
        }
        const PredecessorsMap predecessors()const{
            return predMap_;
        }
    private:

        template<
            class ArcWeights, 
            class SUBGRAPH_MASK,
            class VISITOR 
        >
        void runImpl(
            ArcWeights arcWeights,
            const SUBGRAPH_MASK &  subgraphMask,
            VISITOR && visitor
        ){

            for(auto i=1; i<g_.numberOfNodes(); ++i){
                for(auto arc : g_.arcs()){
                    {
                        const auto s = g_.source(arc);
                        const auto t = g_.target(arc);
                        const auto w =  arcWeights[arc];
                        if(distMap_[s] + w < distMap_[t]){
                            distMap_[t] = distMap_[s] + w;
                            predMap_[t] = s;
                        }
                    }
                }
            }
            // check for negative cycles
            for(auto arc : g_.arcs()){
                const auto s = g_.source(arc);
                const auto t = g_.target(arc);
                if(distMap_[s] + arcWeights[arc] < distMap_[t]){
                    throw CycleWithNegativeWeightsDetectedError();
                }
            }
        }


        template<class SOURCE_ITER>
        void initializeMaps(SOURCE_ITER sourceBegin, SOURCE_ITER sourceEnd){
                
            for(auto node : g_.nodes()){
                predMap_[node] = -1;
                distMap_[node] = std::numeric_limits<WeightType>::infinity();
            }

            for( ; sourceBegin!=sourceEnd; ++sourceBegin){
                auto n = *sourceBegin;
                distMap_[n] = 0;
                predMap_[n] = n;
            }
        }



        const GRAPH & g_;
        PredecessorsMap predMap_;
        DistanceMap     distMap_;
    };



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_SHORTEST_PATH_HXX
