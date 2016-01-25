#pragma once
#ifndef NIFTY_GRAPH_DETAIL_SEARCH_IMPL_HXX
#define NIFTY_GRAPH_DETAIL_SEARCH_IMPL_HXX

#include <queue>
#include <stack>

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/queue/changeable_priority_queue.hxx"

namespace nifty{
namespace graph{
namespace detail_graph{

    template<class T>
    struct LiFo : public std::stack<T>{
        using std::stack<T>::stack;
        const T & nextElement(){
            return this->top();
        }
    };
    template<class T>
    struct FiFo : public std::queue<T>{
        using std::queue<T>::queue;
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
        typedef QUEUE    Queue;
    public:
        SearchImpl(const Graph & g)
        :   g_(g),
            queue_(),
            predMap_(g),
            distMap_(g){
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
                int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return topNode != target;
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
            auto visitor = [](   int64_t topNode,
                const DistanceMap     & distances,
                const PredecessorsMap & predecessors
            ){
                return true;
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

            while(!queue_.empty()){
                auto u = queue_.nextElement();
                queue_.pop();

                // exit on visitors demand
                if(!visitor(u, distMap_, predMap_)){
                    break;
                }

                // if node == gloal node
                //  break
                for(auto adj : g_.adjacency(u)){
                    const auto v = adj.node();
                    const auto e = adj.edge();
                    if(predMap_[v] == -1 &&  subgraphMask.useNode(v) && subgraphMask.useEdge(e)){
                        predMap_[v] = u;
                        distMap_[v] = distMap_[u] + 1;
                        queue_.push(v);
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
