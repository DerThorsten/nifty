#pragma once

#include <functional>
#include <set>
#include <unordered_set>
#include <boost/container/flat_set.hpp>
#include <string>
#include <cmath>        // std::abs

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"
#include <iostream>
#include <algorithm>    // std::max
#include <xtensor/xview.hpp>
using namespace xt::placeholders;


namespace nifty{
    namespace graph{
        namespace agglo{


// UCM: ultra contour map (enable an edge union-find datastructure)
// UPDATE_RULE: the type of linkage criteria implemented in ./details/merge_rules.hxx

            template<
                    class GRAPH, class UPDATE_RULE, bool ENABLE_UCM
            >
            class GaspClusterPolicy{
                typedef GaspClusterPolicy<
                        GRAPH, UPDATE_RULE,  ENABLE_UCM
                > SelfType;

            private:
                typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
                typedef typename GRAPH:: template EdgeMap<float> FloatEdgeMap;
                typedef typename GRAPH:: template NodeMap<float> FloatNodeMap;

                // Flat sets for storing cannot-link constraints:
                typedef boost::container::flat_set<uint64_t> SetType;
                typedef typename GRAPH:: template NodeMap<SetType > NonLinkConstraints;

                typedef UPDATE_RULE UpdateRuleType;
            public:
                typedef typename UpdateRuleType::SettingsType UpdateRuleSettingsType;


                // input types
                typedef GRAPH                                       GraphType;
                typedef FloatEdgeMap                                EdgePrioType;
                typedef FloatEdgeMap                                EdgeSizesType;
                typedef FloatNodeMap                                NodeSizesType;

                struct SettingsType{
                    UpdateRuleSettingsType updateRule;
                    uint64_t numberOfNodesStop{1};
                    double sizeRegularizer{0.};
                    bool addNonLinkConstraints{false};
                    bool mergeConstrainedEdgesAtTheEnd{false};
                    bool collectStats{false};
                };

                enum class EdgeStates : uint8_t {
                    LOCAL = 0, // Edge represent e.g. two direct neighbors that can be contracted
                    LIFTED = 1, // Lifted edges cannot be contracted until it is merged with some local one
                };

                typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

                friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
            private:

                // internal types
                typedef nifty::tools::ChangeablePriorityQueue< float , std::greater<float> > QueueType;


            public:

                template<class SIGNED_WEIGHTS, class IS_LOCAL_EDGE, class EDGE_SIZES, class NODE_SIZES>
                GaspClusterPolicy(const GraphType &,
                                  const SIGNED_WEIGHTS & ,
                                  const IS_LOCAL_EDGE &,
                                  const EDGE_SIZES & ,
                                  const NODE_SIZES & ,
                                  const SettingsType & settings = SettingsType());


                std::pair<uint64_t, double> edgeToContractNext() const;
                bool isDone();

                // callback called by edge contraction graph

                EdgeContractionGraphType & edgeContractionGraph();

            private:
                double pqMergePrio(const uint64_t edge) const;
                double computeWeight(const uint64_t edge) const;

            public:
                // callbacks called by edge contraction graph
                void contractEdge(const uint64_t edgeToContract);
                void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
                void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
                void contractEdgeDone(const uint64_t edgeToContract);

                bool isEdgeConstrained(const uint64_t edge){
                    const auto uv = edgeContractionGraph_.uv(edge);
                    const auto u = uv.first;
                    const auto v = uv.second;
                    const auto & setU  = nonLinkConstraints_[u];
                    const auto & setV  = nonLinkConstraints_[v];
                    // This find operation gets expensive (log(N)), so we look into the smaller set:
                    // NIFTY_CHECK((setU.find(v)!=setU.end()) == (setV.find(u)!=setV.end()),"");
                    if (setU.size() < setV.size()) {
                        return setU.find(v)!=setU.end();
                    } else {
                        return setV.find(u)!=setV.end();
                    }
                }


                bool isMergeAllowed(const uint64_t edge) const{
                    // Here we do not care about the fact that an edge is lifted or not.
                    // We just look at the priority

                    return accumulated_weights_[edge] > 0.;
                }

                double edgeCostInPQ(const uint64_t edge) const{
                    const auto priority = accumulated_weights_[edge];
                    if (settings_.addNonLinkConstraints) {
                        return std::abs(priority);
                    } else {
                        return priority;
                    }
                }

                void addNonLinkConstraint(const uint64_t edge){
                    const auto reprEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
                    NIFTY_ASSERT(accumulated_weights_[reprEdge]<=0.);
                    const auto uv = edgeContractionGraph_.uv(reprEdge);
                    const auto u = uv.first;
                    const auto v = uv.second;
                    // This insert operation, first performs a search (logN) and then insert (if not found)accumulated_weights_[reprEdge]
                    nonLinkConstraints_[uv.first].insert(uv.second);
                    nonLinkConstraints_[uv.second].insert(uv.first);
                }

                auto exportFinalNodeDataOriginalGraph(){
                    // Export node data of the ORIGINAL graph:
                    NIFTY_ASSERT(settings_.collectStats);
                    typename xt::xtensor<float, 2>::shape_type retshape;
                    retshape[0] = graph_.nodeIdUpperBound()+1;
                    retshape[1] = 4;
                    xt::xtensor<float, 2> out(retshape);

                    graph_.forEachNode([&](const uint64_t node) {
                        out(node, 0) = maxNodeSize_per_iter_[node];
                        out(node, 1) = maxCostInPQ_per_iter_[node];
                        out(node, 2) = meanNodeSize_per_iter_[node];
                        out(node, 3) = variance_[node];
                    });
                    return out;
                }

                auto exportFinalEdgeDataContractedGraph(){
                    NIFTY_ASSERT(settings_.collectStats);

                    // Export edge data of the final contracted graph:
                    xt::xtensor<float, 2> out = xt::zeros<float>({uint64_t(edgeContractionGraph().numberOfEdges()), uint64_t(4)});
                    xt::xtensor<bool, 1> notVisitedEdges = xt::ones<bool>({graph_.edgeIdUpperBound()+1});
                    uint64_t edge_counter = 0;
                    graph_.forEachEdge([&](const uint64_t edge) {
                        const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
                        const auto uv = edgeContractionGraph_.uv(cEdge);
                        const auto u = edgeContractionGraph_.findRepresentativeNode(uv.first);
                        const auto v = edgeContractionGraph_.findRepresentativeNode(uv.second);
                        if (u != v && notVisitedEdges(cEdge)) {
                            out(edge_counter, 0) = u;
                            out(edge_counter, 1) = v;
                            out(edge_counter, 2) = accumulated_weights_[cEdge];
                            out(edge_counter, 3) = accumulated_weights_.weight(cEdge);
                            edge_counter++;
                            notVisitedEdges(cEdge) = false;
                        };
                    });

                    // -------
                    // Export data merges/constraints:
                    // -------

                    // First, we collect information/statistics about constraints:
                    //  - sum of constrained edge sizes
                    //  - sum of constrained edge weights
                    //  - max constraint
                    //  - min constraint
                    //  - (collect more stats about size of the edges...?)
                    xt::xtensor<float, 2> constraintStatistics = xt::zeros<float>({uint64_t(graph_.edgeIdUpperBound()+1), uint64_t(6)});
                    graph_.forEachEdge([&](const uint64_t edge){
                        if (constraintsStats_(edge, 0) == 1.) {
                            // Find the ID of the final boundary (could be now merged or not):
                            const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);
                            constraintStatistics(cEdge, 0) += constraintsStats_(edge,2); // Add constraint size
                            constraintStatistics(cEdge, 1) += constraintsStats_(edge,1)*constraintsStats_(edge,2); // Increase sum
                            // The weight of the constrained edge will be always negative, so we keep it in mind while saving the min/max
                            constraintStatistics(cEdge, 2) = std::max(constraintStatistics(cEdge, 2),-constraintsStats_(edge,1));
                            constraintStatistics(cEdge, 3) = std::min(constraintStatistics(cEdge, 3),constraintsStats_(edge,1));
                            // Export stats in case the onstrained edge became positive at some point:
                            constraintStatistics(cEdge, 4) += constraintsStats_(edge,4); // Add constraint size
                            constraintStatistics(cEdge, 5) += constraintsStats_(edge,3)*constraintsStats_(edge,4); // Increase sum
                        }

                    });

                    // Then, we collect information about merged edges and remaining edges in the final graph:
                    //  - was the edge merged or not (1 if yes, 0 if not)
                    //  - edge weight (either before to be merged or in the final graph)
                    //  - edge size (either before to be merged or in the final graph)
                    xt::xtensor<float, 2> mergeStatistics = xt::zeros<float>({uint64_t(graph_.edgeIdUpperBound()+1), uint64_t(4)});
                    graph_.forEachEdge([&](const uint64_t edge){
                        const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(edge);

                        // First, we fill the remaining parts of the constraintStatistics array:
                        constraintStatistics(edge, 0) = constraintStatistics(cEdge, 0);
                        constraintStatistics(edge, 1) = constraintStatistics(cEdge, 1);
                        constraintStatistics(edge, 2) = constraintStatistics(cEdge, 2);
                        constraintStatistics(edge, 3) = constraintStatistics(cEdge, 3);
                        constraintStatistics(edge, 4) = constraintStatistics(cEdge, 4);
                        constraintStatistics(edge, 5) = constraintStatistics(cEdge, 5);

                        // Find the ID of the final boundary (could be now merged or not):
                        const auto uv = edgeContractionGraph_.uv(cEdge);
                        const auto u = edgeContractionGraph_.findRepresentativeNode(uv.first);
                        const auto v = edgeContractionGraph_.findRepresentativeNode(uv.second);
                        if (u != v) {
                            // In this case, the edge was not merged:
                            mergeStatistics(edge, 0) = 0.;
                        } else {
                            // In this case, the edge was merged:
                            mergeStatistics(edge, 0) = 1.;
                        }
                        // Now save the information about the boundary:
                        mergeStatistics(edge, 1) = accumulated_weights_[cEdge];
                        mergeStatistics(edge, 2) = accumulated_weights_.weight(cEdge);
                        mergeStatistics(edge, 3) = mergeStats_(cEdge);
                    });

                    return std::make_tuple(out, constraintStatistics, mergeStatistics);
                }

                auto exportAction(){
                    NIFTY_ASSERT(settings_.collectStats);
                    // Exports actions done
                    // xt::xtensor<float, 2> out = xt::zeros<float>({uint64_t(nb_performed_contractions_+100), uint64_t(5)});
                    xt::xtensor<uint64_t , 2> out = xt::view(actionStats_, xt::all(), xt::all());
                    return out;
                }



            private:



                // INPUT
                const GraphType &   graph_;

                NonLinkConstraints nonLinkConstraints_;


                UPDATE_RULE accumulated_weights_;
                uint64_t nb_performed_contractions_;

                // Stats:
                uint64_t nb_edges_popped_;
                xt::xtensor<uint64_t , 2> actionStats_;
                xt::xtensor<float , 2> constraintsStats_;
                xt::xtensor<float , 1> mergeStats_;


                // State of the edges (LOCAL or LIFTED)
                typename GRAPH:: template EdgeMap<EdgeStates>  edgeState_;

                SettingsType        settings_;

                // Contracted graph:
                EdgeContractionGraphType edgeContractionGraph_;

                // Priority queue:
                QueueType pq_;

                uint64_t edgeToContractNext_;
                double   edgeToContractNextMergePrio_;

                NodeSizesType nodeSizes_;
                NodeSizesType maxNodeSize_per_iter_;
                NodeSizesType meanNodeSize_per_iter_;
                NodeSizesType variance_;
                NodeSizesType maxCostInPQ_per_iter_;
                uint64_t max_node_size_;
                uint64_t sum_node_size_;
                uint64_t quadratic_sum_node_size_;
            };

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            template<class SIGNED_WEIGHTS, class IS_LOCAL_EDGE,class EDGE_SIZES,class NODE_SIZES>
            inline GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            GaspClusterPolicy(
                    const GraphType & graph,
                    const SIGNED_WEIGHTS & signedWeights,
                    const IS_LOCAL_EDGE & isLocalEdge,
                    const EDGE_SIZES      & edgeSizes,
                    const NODE_SIZES      & nodeSizes,
                    const SettingsType & settings
            )
                    :   graph_(graph),
                        nonLinkConstraints_(graph),
                        accumulated_weights_(graph, signedWeights, edgeSizes, settings.updateRule),
                        edgeState_(graph),
                        nodeSizes_(graph),
                        pq_(graph.edgeIdUpperBound()+1),
                        settings_(settings),
                        edgeContractionGraph_(graph, *this),
                        maxNodeSize_per_iter_(graph),
                        maxCostInPQ_per_iter_(graph),
                        variance_(graph),
                        meanNodeSize_per_iter_(graph),
                        max_node_size_(0),
                        sum_node_size_(0),
                        quadratic_sum_node_size_(0),
                        nb_performed_contractions_(0),
                        nb_edges_popped_(0)
            {
                if (settings_.collectStats) {
                    // TODO: update shape actionStats
                    actionStats_ = xt::zeros<uint64_t>({uint64_t(graph_.numberOfNodes()), uint64_t(5)});
                    constraintsStats_ = xt::zeros<float>({uint64_t(graph_.numberOfEdges()), uint64_t(5)});
                    mergeStats_ = xt::zeros<float>({uint64_t(graph_.numberOfEdges())});
                }
                graph_.forEachNode([&](const uint64_t node) {
                    nodeSizes_[node] = nodeSizes[node];
                    sum_node_size_ += nodeSizes[node];
                    quadratic_sum_node_size_ += nodeSizes[node] * nodeSizes[node];
                    if (nodeSizes[node] > max_node_size_)
                        max_node_size_ = uint8_t(nodeSizes[node]);
                });

                graph_.forEachEdge([&](const uint64_t edge){
                    const auto loc = isLocalEdge[edge];
                    edgeState_[edge] = (loc == 1 ? EdgeStates::LOCAL : EdgeStates::LIFTED);
                    pq_.push(edge, this->computeWeight(edge));
                });
            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline std::pair<uint64_t, double>
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            edgeToContractNext() const {
                return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline bool
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::isDone(
            ){
                while(true) {
                    while(!pq_.empty() && !isNegativeInf(pq_.topPriority()) && edgeContractionGraph_.numberOfNodes() > settings_.numberOfNodesStop){
                        // Here we already know that the edge is not lifted
                        // (Otherwise we would have inf cost in PQ)
                        const auto nextActioneEdge = pq_.top();

                        if (settings_.collectStats) {
                            // TODO: we can have more than nb_graph_edges popped from PQ, so old code could give seg fault
                            // nb_edges_popped_++;
                            // actionStats_(nb_edges_popped, 0) = graph_.uv(nextActioneEdge).first;
                            // actionStats_(nb_edges_popped, 1) = graph_.uv(nextActioneEdge).second;
                            // actionStats_(nb_edges_popped, 2) = nextActioneEdge;
                            // actionStats_(nb_edges_popped, 3) = pq_.topPriority();
                        }

                        // Check if some early constraints were enforced:
                        if (settings_.addNonLinkConstraints) {
                            if(this->isEdgeConstrained(nextActioneEdge)) {
                                if (settings_.collectStats) {
                                    // Check if the constrained edge is now positive and in case remember about it:
                                    if (this->isMergeAllowed(nextActioneEdge)) {
                                        const auto reprEdge = edgeContractionGraph_.findRepresentativeEdge(
                                                nextActioneEdge);
                                        constraintsStats_(reprEdge, 3) = accumulated_weights_[reprEdge];
                                        constraintsStats_(reprEdge, 4) = accumulated_weights_.weight(reprEdge);
                                    }
                                    // Remember that an edge was constrained at this iteration:
                                    // actionStats_(nb_edges_popped_, 4) = 1;
                                }
                                pq_.push(nextActioneEdge, -1.0*std::numeric_limits<double>::infinity());
                                continue;
                            }
                        }

                        // Here we check if we are allowed to make the merge:
                        if(this->isMergeAllowed(nextActioneEdge)){
                            edgeToContractNext_ = nextActioneEdge;
                            edgeToContractNextMergePrio_ = pq_.topPriority();
                            if (settings_.collectStats) {
                                // actionStats_(nb_performed_contractions_, 0) = graph_.uv(nextActioneEdge).first;
                                // actionStats_(nb_performed_contractions_, 1) = graph_.uv(nextActioneEdge).second;
                                // actionStats_(nb_performed_contractions_, 2) = nextActioneEdge;
                                // actionStats_(nb_performed_contractions_, 3) = pq_.topPriority();
                                // actionStats_(nb_performed_contractions_, 4) = 1;
                            }
                            return false;
                        }
                        else{
                            if (! settings_.addNonLinkConstraints) {
                                // In this case we know that we reached priority zero, so we can already stop
                                return true;
                            }
                            this->addNonLinkConstraint(nextActioneEdge);
                            pq_.push(nextActioneEdge, -1.0*std::numeric_limits<double>::infinity());
                            if (settings_.collectStats) {
                                // actionStats_(nb_edges_popped_, 4) = 2;
                            }

                            // Remember about weight and size of the constrained edge:
                            if (settings_.collectStats) {
                                const auto reprEdge = edgeContractionGraph_.findRepresentativeEdge(nextActioneEdge);
                                constraintsStats_(reprEdge, 0) = 1.;
                                constraintsStats_(reprEdge, 1) = accumulated_weights_[reprEdge]; // Remember the edge weight
                                constraintsStats_(reprEdge, 2) = accumulated_weights_.weight(reprEdge); // Remember the edge size
                            }
                        }
                    }
                    if (settings_.addNonLinkConstraints && settings_.mergeConstrainedEdgesAtTheEnd) {
                        // We push again all values to PQ and merge what is left and positive:
                        settings_.addNonLinkConstraints = false;
                        graph_.forEachEdge([&](const uint64_t e){
                            const auto cEdge = edgeContractionGraph_.findRepresentativeEdge(e);
                            const auto uv = edgeContractionGraph_.uv(cEdge);
                            const auto u = edgeContractionGraph_.findRepresentativeNode(uv.first);
                            const auto v = edgeContractionGraph_.findRepresentativeNode(uv.second);
                            if (u != v) {
                                pq_.push(cEdge, this->computeWeight(cEdge));
                            }
                        });
                        continue;
                    }
                    return true;
                }
            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline double
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            pqMergePrio(
                    const uint64_t edge
            ) const {
                const auto s = edgeState_[edge];
                double costInPQ;
                if(s == EdgeStates::LOCAL){
                    costInPQ = this->edgeCostInPQ(edge);
                }
                else{
                    // In this case the edge is lifted, so we need to be careful.
                    // It can be inserted in the PQ to constrain, but not to merge.
                    // REMARK: The second condition  "isMergeAllowed" is actually not necessary, because it is anyway checked
                    // again in isDone before to actually contract the edge...
                    if (!settings_.addNonLinkConstraints || this->isMergeAllowed(edge))
                        costInPQ = -1.0*std::numeric_limits<double>::infinity();
                    else {
                        costInPQ = this->edgeCostInPQ(edge);
                    }
                }
                return costInPQ;
            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline void
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            contractEdge(
                    const uint64_t edgeToContract
            ){
                // Remember about the highest cost in PQ:
                if (settings_.collectStats) {
                    mergeStats_(edgeToContract) = nb_performed_contractions_;
                    actionStats_(nb_performed_contractions_, 0) = edgeContractionGraph_.uv(edgeToContract).first;
                    actionStats_(nb_performed_contractions_, 1) = edgeContractionGraph_.uv(edgeToContract).second;
                    actionStats_(nb_performed_contractions_, 2) = edgeToContract;
                    actionStats_(nb_performed_contractions_, 3) = 1;
                    // actionStats_(nb_performed_contractions_, 3) = pq_.topPriority();

                }
                maxCostInPQ_per_iter_[nb_performed_contractions_] = this->edgeCostInPQ(edgeToContract);
                pq_.deleteItem(edgeToContract);
            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline typename GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::EdgeContractionGraphType &
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            edgeContractionGraph(){
                return edgeContractionGraph_;
            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline void
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            mergeNodes(
                    const uint64_t aliveNode,
                    const uint64_t deadNode
            ){
                // Save data about max_node_size
                maxNodeSize_per_iter_[nb_performed_contractions_] = max_node_size_;
                const auto remaining_nodes = edgeContractionGraph_.numberOfNodes();
                meanNodeSize_per_iter_[nb_performed_contractions_] = float(sum_node_size_) / float(remaining_nodes);
                variance_[nb_performed_contractions_] =  float(quadratic_sum_node_size_) / float(remaining_nodes) - std::pow(meanNodeSize_per_iter_[nb_performed_contractions_], 2);

                quadratic_sum_node_size_ += std::pow(nodeSizes_[deadNode] + nodeSizes_[aliveNode], 2) - std::pow(nodeSizes_[aliveNode], 2) - std::pow(nodeSizes_[deadNode], 2);

                nodeSizes_[aliveNode] += nodeSizes_[deadNode];
                if (nodeSizes_[aliveNode] > max_node_size_)
                    max_node_size_ = uint64_t(nodeSizes_[aliveNode]);


                if (settings_.addNonLinkConstraints) {
                    auto  & aliveNodeNlc = nonLinkConstraints_[aliveNode];
                    const auto & deadNodeNlc = nonLinkConstraints_[deadNode];
                    aliveNodeNlc.insert(deadNodeNlc.begin(), deadNodeNlc.end());


                    for(const auto v : deadNodeNlc){
                        auto & nlc = nonLinkConstraints_[v];

                        // best way to change values in set...
                        nlc.erase(deadNode);
                        nlc.insert(aliveNode);
                    }

                    aliveNodeNlc.erase(deadNode);
                }

            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline void
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            mergeEdges(
                    const uint64_t aliveEdge,
                    const uint64_t deadEdge
            ){

                NIFTY_ASSERT_OP(aliveEdge,!=,deadEdge);
                NIFTY_ASSERT(pq_.contains(aliveEdge));
                NIFTY_ASSERT(pq_.contains(deadEdge));

                pq_.deleteItem(deadEdge);

                // update priority:
                accumulated_weights_.merge(aliveEdge, deadEdge);


                // update state
                // const auto oldStateAlive = edgeState_[aliveEdge];
                auto & sa = edgeState_[aliveEdge];
                const auto  sd = edgeState_[deadEdge];
                if(
                        sa == EdgeStates::LOCAL || sd == EdgeStates::LOCAL
                        ){
                    sa = EdgeStates::LOCAL;
                }
                else if( sa == EdgeStates::LIFTED  && sd == EdgeStates::LIFTED )
                {
                    sa = EdgeStates::LIFTED;
                }

                const auto sr = settings_.sizeRegularizer;
                bool add_to_PQ = true;
                if (settings_.addNonLinkConstraints) {
                    if (this->isEdgeConstrained(aliveEdge)) {
                        // If the edge was already constrained, it does not make sense to add it again to PQ
                        add_to_PQ = false;
                    }
                }
                if (sr > 0.00001)
                    add_to_PQ = false;
                if (add_to_PQ)
                    pq_.push(aliveEdge, this->computeWeight(aliveEdge));

            }


            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline void
            GaspClusterPolicy<GRAPH, UPDATE_RULE, ENABLE_UCM>::
            contractEdgeDone(
                    const uint64_t edgeToContract
            ){
                // HERE WE UPDATE the PQ when a SizeReg is used:
                const auto sr = settings_.sizeRegularizer;
                if (sr > 0.000001) {
                    const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
                    for(auto adj : edgeContractionGraph_.adjacency(u)){
                        const auto edge = adj.edge();
                        pq_.push(edge, computeWeight(edge));
                    }
                }

                nb_performed_contractions_++;

            }

            template<class GRAPH, class UPDATE_RULE, bool ENABLE_UCM>
            inline double
            GaspClusterPolicy<GRAPH, UPDATE_RULE,  ENABLE_UCM>::
            computeWeight(
                    const uint64_t edge
            ) const {
                const auto fromEdge = this->pqMergePrio(edge); // This -inf if the edge was lifted
                const auto sr = settings_.sizeRegularizer;

                if (sr > 0.000001 && !isNegativeInf(fromEdge))
                {
                    const auto uv = edgeContractionGraph_.uv(edge);
                    const auto sizeU = nodeSizes_[uv.first];
                    const auto sizeV = nodeSizes_[uv.second];
                    const auto sFac = 2.0 / ( 1.0/std::pow(sizeU,sr) + 1.0/std::pow(sizeV,sr) );
                    return fromEdge * (1. / sFac);
                } else {
                    return fromEdge;
                }
            }


        } // namespace agglo
    } // namespace nifty::graph
} // namespace nifty

