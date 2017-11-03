#pragma once

#include <cstddef>


#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{









    template<class CHILD_OBJECTIVE, class GRAPH, class LIFTED_GRAPH, class WEIGHT_TYPE>
    class LiftedMulticutObjectiveBase{
    public:

        typedef CHILD_OBJECTIVE ChildObjective;
        typedef LiftedMulticutObjectiveBase<ChildObjective, GRAPH, LIFTED_GRAPH, WEIGHT_TYPE> Self;


        template<class NODE_LABELS>
        WEIGHT_TYPE evalNodeLabels(const NODE_LABELS & nodeLabels)const{
            WEIGHT_TYPE sum = static_cast<WEIGHT_TYPE>(0.0);

            const auto & w = _child().weights();
            const auto & lg = _child().liftedGraph();

            for(const auto edge: lg.edges()){
                const auto uv = lg.uv(edge);

                if(nodeLabels[uv.first] != nodeLabels[uv.second]){
                    sum += w[edge];
                }
            }
            return sum;
        }

        uint64_t numberOfLiftedEdges()const{
            return _child().liftedGraph().numberOfEdges() - _child().graph().numberOfEdges();
        }


    private:
        ChildObjective & _child(){
           return *static_cast<ChildObjective *>(this);
        }
        const ChildObjective & _child()const{
           return *static_cast<const ChildObjective *>(this);
        }

    };





    template<class GRAPH, class WEIGHT_TYPE>
    class LiftedMulticutObjective :  public
        LiftedMulticutObjectiveBase<
            LiftedMulticutObjective<GRAPH, WEIGHT_TYPE>,
            GRAPH, UndirectedGraph<>, WEIGHT_TYPE
        >
    {
    private:
        //typedef nifty::graph::detail_graph::NodeIndicesToContiguousNodeIndices<GRAPH > ToContiguousNodes;


        typedef std::is_same<typename GRAPH::NodeIdTag,  ContiguousTag> GraphHasContiguousNodeIds;

        static_assert( GraphHasContiguousNodeIds::value,
                  "LiftedMulticut assumes that the node id-s between graph and lifted graph are exchangeable \
                   The LiftedMulticutObjective can only guarantee this for for graphs which have Contiguous Node ids "
        );

    public:
        typedef GRAPH GraphType;








        typedef UndirectedGraph<> LiftedGraphType;

        typedef typename GraphType:: template NodeMap<uint64_t> NodeLabelsType;

        typedef LiftedGraphType LiftedGraph;

        typedef WEIGHT_TYPE WeightType;
        typedef graph_maps::EdgeMap<LiftedGraph, WeightType> WeightsMapType;
        typedef WeightsMapType WeightsMap;


        LiftedMulticutObjective(const GraphType & graph, const int64_t reserveAdditionalEdges = -1)
        :   graph_(graph),
            liftedGraph_(graph.numberOfNodes(), graph.numberOfEdges() + (reserveAdditionalEdges<0 ?  graph.numberOfEdges() : reserveAdditionalEdges) ),
            weights_(liftedGraph_,0){

            for(const auto edge : graph_.edges()){
                const auto uv = graph_.uv(edge);
                liftedGraph_.insertEdge(
                    uv.first,
                    uv.second
                );
            }
            NIFTY_CHECK_OP(liftedGraph_.numberOfEdges(), == , graph_.numberOfEdges(),"");
            weights_.insertedEdges(liftedGraph_.edgeIdUpperBound(),0);
        }

        std::pair<bool,uint64_t> setCost(const uint64_t u, const uint64_t v, const WeightType & w = 0.0, const bool overwrite = false){
            const auto preSize = liftedGraph_.numberOfEdges();
            const auto edge = liftedGraph_.insertEdge(u,v);
            if( liftedGraph_.numberOfEdges() > preSize){
                weights_.insertedEdges(edge, w);
                return std::pair<bool,uint64_t>(edge,true);
            }
            else{
                if(overwrite)
                    weights_[edge] = w;
                else
                    weights_[edge] += w;
                return std::pair<bool,uint64_t>(edge,false);
            }
        }

        WeightsMap & weights(){
            return weights_;
        }
        const WeightsMap & weights() const{
            return weights_;
        }

        const GraphType & graph() const{
            return graph_;
        }

        const LiftedGraph & liftedGraph() const{
            return liftedGraph_;
        }


        void insertLiftedEdgesBfs(const std::size_t maxDistance){

            BreadthFirstSearch<GraphType> bfs(graph_);
            graph_.forEachNode([&](const uint64_t sourceNode){
                bfs.graphNeighbourhood(sourceNode, maxDistance, [&](const uint64_t targetNode, const uint64_t ){
                    this->setCost(sourceNode, targetNode, 0.0);
                });
            });
        }

        template<class DIST_VEC_TYPE>
        void insertLiftedEdgesBfs(const std::size_t maxDistance, DIST_VEC_TYPE & distVec){

            BreadthFirstSearch<GraphType> bfs(graph_);
            graph_.forEachNode([&](const uint64_t sourceNode){
                bfs.graphNeighbourhood(sourceNode, maxDistance, [&](const uint64_t targetNode, const uint64_t dist){
                    if(this->setCost(sourceNode, targetNode, 0.0).second){
                        distVec.push_back(dist);
                    }
                });
            });
        }

        int64_t graphEdgeInLiftedGraph(const uint64_t graphEdge)const{

            typedef std::is_same<typename GraphType::EdgeIdTag,  ContiguousTag> CondA;
            typedef std::is_same<typename GraphType::EdgeIdOrderTag, SortedTag> CondB;

            if(CondA::value && CondB::value  ){
                return graphEdge;
            }
            else{
                // this is not efficient, we should refactor this
                const auto uv = graph_.uv(graphEdge);
                return liftedGraph_.findEdge(uv.first, uv.second);
            }
        }

        int64_t liftedGraphEdgeInGraph(const uint64_t liftedGraphEdge)const{

            typedef std::is_same<typename GraphType::EdgeIdTag,  ContiguousTag> CondA;
            typedef std::is_same<typename GraphType::EdgeIdOrderTag, SortedTag> CondB;

            if(CondA::value && CondB::value  ){
                if(liftedGraphEdge < graph_.numberOfEdges())
                    return liftedGraphEdge;
                else
                    return -1;
            }
            else{
                // this is not efficient, we should refactor this
                const auto uv = liftedGraph_.uv(liftedGraphEdge);
                return graph_.findEdge(uv.first, uv.second);
            }
        }

        /**
         * @brief Iterate over all edges of the lifted graph which are in the original graph
         * @details Iterate over all edges of the lifted graph which are in the original graph.
         * The ids are w.r.t. the lifted graph
         *
         * @param f functor/lambda which is called for each edge id
         */
        template<class F>
        void forEachGraphEdge(F && f)const{
            for(uint64_t e = 0 ; e<graph_.numberOfEdges(); ++e){
                f(e);
            }
        }


        template<class F>
        void parallelForEachGraphEdge(
            parallel::ThreadPool & threadpool,
            F && f
        )const{
            parallel::parallel_foreach(threadpool,graph_.numberOfEdges(),
            [&](const int tid, const uint64_t e){
                f(tid, e);
            });
        }


        /**
         * @brief Iterate over all edges of the lifted graph which are NOT in the original graph.
         * @details Iterate over all edges of the lifted graph which are NOT the original graph.
         * The ids are w.r.t. the lifted graph
         *
         * @param f functor/lambda which is called for each edge id
         */
        template<class F>
        void forEachLiftedeEdge(F && f)const{
            for(uint64_t e = graph_.numberOfEdges(); e<liftedGraph_.numberOfEdges(); ++e){
                f(e);
            }
        }

        template<class F>
        void parallelForEachLiftedeEdge(
            parallel::ThreadPool & threadpool,
            F && f
        )const{

            const auto gEdgeNum =  graph_.numberOfEdges();
            parallel::parallel_foreach(threadpool,this->numberOfLiftedEdges(),
            [&](const int tid, const uint64_t i){
                const uint64_t e = i + gEdgeNum;
                f(tid, e);
            });
        }


    protected:
        const GraphType & graph_;
        LiftedGraph liftedGraph_;
        WeightsMap weights_;

    };

} // namespace lifted_multicut
} // namespace nifty::graph::opt
} // namespace nifty::graph
} // namespace nifty
