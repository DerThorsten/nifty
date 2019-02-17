#pragma once

#include <vector>

// for lifted ucm features
#include "nifty/graph/agglo/agglomerative_clustering.hxx"
#include "nifty/graph/agglo/cluster_policies/lifted_graph_edge_weighted_cluster_policy.hxx"

// for shortest path features
#include  "nifty/graph/shortest_path_dijkstra.hxx"


namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{



    template<
        class LIFTED_MULTICUT_OBJECTIVE,
        class EDGE_INDICATORS,
        class EDGE_SIZES,
        class NODE_SIZES,
        class OUT
    >
    void liftedUcmFeatures(
        const LIFTED_MULTICUT_OBJECTIVE & objective,
        const EDGE_INDICATORS & edgeIndicators,
        const EDGE_SIZES & edgeSizes,
        const NODE_SIZES & nodeSizes,
        std::vector<double> sizeRegularizers,
        OUT & out
    ){

        typedef typename LIFTED_MULTICUT_OBJECTIVE::LiftedGraphType LiftedGraphType;
        typedef typename LiftedGraphType:: template EdgeMap<double>  NodeMapDouble;
        typedef typename LiftedGraphType:: template EdgeMap<double>  EdgeMapDouble;
        typedef typename LiftedGraphType:: template EdgeMap<bool>    EdgeMapBool;


        typedef agglo::LiftedGraphEdgeWeightedClusterPolicy<
            LiftedGraphType,
            EdgeMapDouble & ,
            EdgeMapDouble & ,
            NodeMapDouble & ,
            EdgeMapBool   & ,
            true
        > ClusterPolicyType;

        typedef typename ClusterPolicyType::SettingsType ClusterPolicySettingsType;
        typedef agglo::AgglomerativeClustering<ClusterPolicyType> AgglomerativeClusteringType;


        ClusterPolicySettingsType settings;
        settings.numberOfNodesStop = 1;
        settings.numberOfEdgesStop = 0;

        const auto & graph = objective.graph();
        const auto & liftedGraph = objective.liftedGraph();


        EdgeMapDouble lgEdgeIndicators(liftedGraph);
        EdgeMapDouble lgEdgeSizes(liftedGraph);
        NodeMapDouble lgNodeSizes(liftedGraph);
        EdgeMapBool   lgEdgeIsLifted(liftedGraph);
        EdgeMapDouble lgOutBuffer(liftedGraph);

        auto fi = 0;
        for(std::size_t i=0; i<sizeRegularizers.size(); ++i){

            // reset / fill maps
            graph.forEachEdge([&](const uint64_t graphEdge){
                const auto liftedEdge = objective.graphEdgeInLiftedGraph(graphEdge);
                lgEdgeIndicators[liftedEdge] = edgeIndicators[graphEdge];
                lgEdgeSizes[liftedEdge] = edgeSizes[graphEdge];
                lgEdgeIsLifted[liftedEdge] = false;
            });

            objective.forEachLiftedeEdge([&](const uint64_t liftedEdge){
                lgEdgeIndicators[liftedEdge] = 0.0;
                lgEdgeSizes[liftedEdge] = 0.0;
                lgEdgeIsLifted[liftedEdge] = true;
            });

            liftedGraph.forEachNode([&](const uint64_t node){
                lgNodeSizes[node] = nodeSizes[node];
            });

 
            settings.sizeRegularizer = sizeRegularizers[i];
            ClusterPolicyType clusterPolicy(liftedGraph, lgEdgeIndicators,
                                            lgEdgeSizes, lgNodeSizes, 
                                            lgEdgeIsLifted, settings);

            AgglomerativeClusteringType agglomerativeClustering(clusterPolicy);


            
            agglomerativeClustering.runAndGetDendrogramHeight(lgOutBuffer);
            auto index = 0;
            objective.forEachLiftedeEdge([&](const uint64_t liftedEdge){
                out(fi, index) = lgOutBuffer[liftedEdge];
                ++index;
            });
            ++fi;

            agglomerativeClustering.ucmTransform(lgEdgeIndicators, lgOutBuffer);
            index = 0;
            objective.forEachLiftedeEdge([&](const uint64_t liftedEdge){
                out(fi, index) = lgOutBuffer[liftedEdge];
                ++index;
            });
            ++fi;
        }
    }



    template<
        class LIFTED_MULTICUT_OBJECTIVE,
        class EDGE_INDICATORS,
        class EDGE_SIZES,
        class NODE_SIZES,
        class OUT
    >
    void liftedShortedPathFeatures(
        const LIFTED_MULTICUT_OBJECTIVE & objective,
        const EDGE_INDICATORS & edgeIndicators,
        std::vector<double> offsets,
        OUT & out,
        const int nThreads = -1
    ){
        typedef typename LIFTED_MULTICUT_OBJECTIVE::GraphType GraphType;
        typedef ShortestPathDijkstra<GraphType, double> ShortestPathType;

        // shortcuts
        const auto & graph = objective.graph();
        const auto & liftedGraph = objective.liftedGraph();

        // threadpool
        parallel::ParallelOptions parallelOptions(nThreads);
        const auto numberOfThreads = parallelOptions.getActualNumThreads();
        parallel::ThreadPool threadpool(parallelOptions);

        // data for each thread
        struct ThreadData{
            ThreadData(const GraphType & g)
            :   shortestPath(g){
            }
            ShortestPathType shortestPath;
        };
        std::vector<ThreadData * > perThreadData(numberOfThreads);
        parallel_foreach(threadpool, numberOfThreads,[&](const int tid, const int i){
            perThreadData[i] = new ThreadData(graph);
        });

        // weight offset
        struct WeightsPlusOffset{
            WeightsPlusOffset(
                const EDGE_INDICATORS & ei,
                const double offset
            )
            :   edgeIndicators_(ei),
                offset_(offset){
            }
            double operator[](const uint64_t edge){
                return edgeIndicators_[edge] + offset_;
            }
            const EDGE_INDICATORS & edgeIndicators_;
            const double offset_;
        };

        for(std::size_t i=0; i<offsets.size(); ++i){

            // run the shortest path algorithm
            objective.parallelForEachLiftedeEdge(threadpool,
            [&](const int tid, const uint64_t liftedEdge){

                auto & threadData = *(perThreadData[tid]);
                auto & sp = threadData.shortestPath;
                const auto uv = liftedGraph.uv();
 
                // run the shortest path alg
                const WeightsPlusOffset weights(edgeIndicators, offsets[i]);
                sp.runSingleSourceSingleTarget(weights, uv.first, uv.second);

                // extract the features form the shortest path

            });
        }

        // delete data for each thread
        parallel_foreach(threadpool, numberOfThreads,[&](const int tid, const int i){
            delete perThreadData[i];
        });

    }



} // namespace lifted_multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty

