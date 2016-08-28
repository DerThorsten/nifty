#pragma once
#ifndef NIFTY_GRAPH_DETAIL_SEARCH_IMPL_HXX
#define NIFTY_GRAPH_DETAIL_SEARCH_IMPL_HXX

#include <queue>
#include <stack>

#include "nifty/graph/subgraph_mask.hxx"


namespace nifty{
namespace graph{
namespace detail_graph{

    template<class T>
    struct LiFo : public std::stack<T>{
        // CLANG DOES NOT LIKE THESE
        //using std::stack<T>::stack;
        const T & nextElement(){
            return this->top();
        }
    };
    template<class T>
    struct FiFo : public std::queue<T>{
        // CLANG DOES NOT LIKE THESE
        //using std::queue<T>::queue;
        const T & nextElement(){
            return this->front();
        }
    };

    template<class GRAPH,class QUEUE>
    class SearchImpl{

    public:
        typedef GRAPH Graph;
        typedef typename Graph:: template NodeMap<int64_t>     PredecessorsMap;
        typedef typename Graph:: template NodeMap<int64_t>  DistanceMap;
    private:
        typedef QUEUE Queue;
    public:
        SearchImpl(const Graph & g)
        :   g_(g),
            queue_(),
            predMap_(g),
            distMap_(g){
        }


        template< class F>
        void graphNeighbourhood(            
            const uint64_t source,
            const size_t maxDistance,
            F && f
        ){

            auto visitor = [&]
            (   
                int64_t toNode,
                int64_t predecessorNode,
                int64_t edge,
                int64_t distance,
                bool & continueSeach,
                bool & addToNode
            ){
                if(distance > maxDistance){
                    addToNode = false;
                }
                else{
                    f(toNode, distance);
                }
            };


            // subgraph mask
            DefaultSubgraphMask<Graph> subgraphMask;

            this->run(&source, &source + 1 , subgraphMask, visitor);
        }




        // run single source single target
        // no  callback no mask exposed
        void runSingleSourceSingleTarget(
            const int64_t source,
            const int64_t target = -1
        ){
            // subgraph mask
            DefaultSubgraphMask<Graph> subgraphMask;
            // visitor
            auto visitor = [&]
            (   
                int64_t toNode,
                int64_t predecessorNode,
                int64_t edge,
                int64_t distance,
                bool & continueSeach,
                bool & addToNode
            ){
                continueSeach =  (toNode != target);
                addToNode = true;
            };

            this->initializeMaps(&source, &source +1);
            runImpl(subgraphMask, visitor);
        }

        template<class SUBGRAPH_MASK>
        void runSingleSourceSingleTarget(
            const int64_t source,
            const int64_t target,
            const SUBGRAPH_MASK & subgraphMask
        ){

            // visitor
            auto visitor = [&]
            (   
                int64_t toNode,
                int64_t predecessorNode,
                int64_t edge,
                int64_t distance,
                bool & continueSeach,
                bool & addToNode
            ){
                continueSeach =  (toNode != target);
                addToNode = true;
            };

            this->initializeMaps(&source, &source +1);
            runImpl(subgraphMask, visitor);
        }


        // run single source  ALL targets
        // no  callback no mask exposed
        void runSingleSource(
            const int64_t source
        ){

            // subgraph mask
            DefaultSubgraphMask<Graph> subgraphMask;
            this->initializeMaps(&source, &source +1);
            // visitor
            auto visitor = [&]
            (   
                int64_t toNode,
                int64_t predecessorNode,
                int64_t edge,
                int64_t distance,
                bool & continueSeach,
                bool & addToNode
            ){
                // algorithm has these initialized to true
                //continueSeach =  true;
                //addToNode = true;
            };
            runImpl(subgraphMask, visitor);
        }

        template<class SOURCE_ITER, class SUBGRAPH_MASK, class VISITOR>
        void run(
            SOURCE_ITER sourceBegin, 
            SOURCE_ITER sourceEnd,
            const SUBGRAPH_MASK &  subgraphMask,
            VISITOR && visitor
        ){
            this->initializeMaps(sourceBegin, sourceEnd);
            this->runImpl(subgraphMask,visitor);
        }

        const DistanceMap & distances()const{
            return distMap_;
        }
        const PredecessorsMap predecessors()const{
            return predMap_;
        }
    private:

        template<
            class SUBGRAPH_MASK,
            class VISITOR 
        >
        void runImpl(
            const SUBGRAPH_MASK &  subgraphMask,
            VISITOR && visitor
        ){
            auto continueSeach = true;
            while(continueSeach && !queue_.empty()){
                auto u = queue_.nextElement();
                queue_.pop();
                // if node == gloal node
                //  break
                for(auto adj : g_.adjacency(u)){
                    const auto v = adj.node();
                    const auto e = adj.edge();
                    if(predMap_[v] == -1 &&  subgraphMask.useNode(v) && subgraphMask.useEdge(e)){

                        const auto newDistance = distMap_[u] + 1;
                        auto addToNode = true;
                        visitor(v,u,e,newDistance,continueSeach, addToNode);
                        if(addToNode){
                            predMap_[v] = u;
                            distMap_[v] = newDistance;
                            queue_.push(v);
                        }
                        if(!continueSeach)
                            break;                        
                    }
                }
            }
        }

        template<class SOURCE_ITER>
        void initializeMaps(SOURCE_ITER sourceBegin, SOURCE_ITER sourceEnd){
            while(!queue_.empty())
                queue_.pop();
            for(auto node : g_.nodes()){
                predMap_[node] = -1;
            }

            for( ; sourceBegin!=sourceEnd; ++sourceBegin){
                auto n = *sourceBegin;
                distMap_[n] = 0;
                predMap_[n] = n;
                queue_.push(n);
            }
        }

        const GRAPH & g_;
        Queue queue_;
        PredecessorsMap predMap_;
        DistanceMap     distMap_;
    };

} // namespace nifty::graph::detail_graph
} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_DETAIL_SEARCH_IMPL_HXX
