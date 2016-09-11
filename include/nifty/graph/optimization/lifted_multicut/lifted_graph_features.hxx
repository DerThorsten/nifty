#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_GRAPH_FEATURES_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_GRAPH_FEATURES_HXX

#include <vector>

#include "nifty/graph/agglo/agglomerative_clustering.hxx"
#include "nifty/graph/agglo/cluster_policies/lifted_graph_edge_weighted_cluster_policy.hxx"


namespace nifty{
namespace graph{
namespace lifted_multicut{

    // \cond SUPPRESS_DOXYGEN
    namespace detail_lifted_graph_features{

        // hack to add [] brackets to marray
        template<class ARRAY>
        struct AddBrackets{

            typedef typename  ARRAY::reference reference;
            typedef typename  ARRAY::const_reference const_reference;

            AddBrackets(ARRAY & array)
            : array_(array){

            }

            reference operator[](const size_t i){
                return array_(i);
            } 
            const_reference operator[](const size_t i)const{
                return array_(i);
            } 

            ARRAY & array_;
        };
    };
    // \endcond 


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

        typedef detail_lifted_graph_features::AddBrackets<marray::View<double> > AddBracketsHack;

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

        typedef typename ClusterPolicyType::Settings ClusterPolicySettingsType;
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
        for(size_t i=0; i<sizeRegularizers.size(); ++i){

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




} // namespace lifted_multicut
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_GRAPH_FEATURES_HXX
