#pragma once

#include <cstddef>
#include <limits> // std::numeric_limits
#include <deque>
#include <queue>
#include <vector>
#include <algorithm> // std::reverse
                     
#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/detail/search_impl.hxx"


namespace nifty{
namespace graph{

    // \cond SUPPRESS_DOXYGEN
    namespace detail_bfs{

        template<class T, class DEQUEUE_TYPE>
        inline void
        singleSourceSingleTargetHelper(
            const std::vector<int64_t>& parents,
            const T vPositive,
            const T vNegative,
            DEQUEUE_TYPE& path
        ) {
            //NIFTY_ASSERT_OP(vPositive,>= ,0,"");
            //NIFTY_ASSERT_OP(vNegative,>= ,0,"");
            T t = vPositive;
            for(;;) {
                path.push_front(t);
                if(parents[t] - 1 == t) {
                    break;
                }
                else {
                    t = parents[t] - 1;
                }
            }
            t = vNegative;
            for(;;) {
                path.push_back(t);
                if(-parents[t] - 1 == t) {
                    break;
                }
                else {
                    t = -parents[t] - 1;
                }
            }
        }
    };  // end namespace detail bfs
    // \cond SUPPRESS_DOXYGEN

    // \cond SUPPRESS_DOXYGEN
    template<class T>
    class RestrictedDeque{
    public:


        void push_front(const T & val){
            frontVec_.push_back(val);
        }
        void push_back(const T & val){
            backVec_.push_back(val);
        }
        void clear(){
            frontVec_.resize(0);
            backVec_.resize(0);
        }

        const T & operator[](const size_t i) const{
            if(i < frontVec_.size()){
                const auto j = frontVec_.size() - 1 - i;
            }
            else{
                const auto j = i - frontVec_.size();
            }
        }
        
        size_t size()const{
            return frontVec_ + backVec_;
        }
    private:
        std::vector<T> frontVec_;
        std::vector<T> backVec_;
    };
    // \endcond



    template<class GRAPH>
    class BidirectionalBreadthFirstSearch{
    private:
        enum RetFlag{
            ReturnTrue,
            ReturnFalse,
            ContinueSeach
        };
        typedef std::deque<uint64_t> DequeType; //
    public:
        typedef GRAPH GraphType;
        typedef typename GraphType:: template NodeMap<int64_t> Parents;

        BidirectionalBreadthFirstSearch(const GraphType & graph)
        :   graph_(graph),
            parents_(graph){

        }


        bool runSingleSourceSingleTarget(
            const int64_t source,
            const int64_t target
        ){
            DefaultSubgraphMask<GraphType> subgraphMask;
            return this->runSingleSourceSingleTarget(source,target,subgraphMask);
        }

        template<class SUBGRAPH_MASK>
        bool runSingleSourceSingleTarget(
            const int64_t source,
            const int64_t target,
            const SUBGRAPH_MASK & mask
        ){
            path_.clear();
            if(!mask.useNode(source) || !mask.useNode(target))
                return false;
            if(source == target){
                path_.push_back(source);
                return true;
            }
            std::fill(parents_.begin(), parents_.end(), 0);
            parents_[source] = source + 1;
            parents_[target] =  -static_cast<int64_t>(target) - 1;

            // clear queues
            for(auto q=0; q<2; ++q){
                while(!queues_[q].empty()){
                    queues_[q].pop();
                }
            }

            queues_[0].push(source);
            queues_[1].push(target);


            for(auto q = 0; true; q = 1 - q) { // infinite loop, alternating queues
                const auto numberOfNodesAtFront = queues_[q].size();
                //std::cout<<"numberOfNodesAtFront q="<<q<<" "<<numberOfNodesAtFront<<"\n";
                for(auto n = 0; n < numberOfNodesAtFront; ++n) {
                    //std::cout<<"n="<<n<<"\n";
                    auto v = queues_[q].front();
                    queues_[q].pop();


                    auto fHelper = [&](const uint64_t otherNode, const uint64_t edge){
                        //std::cout<<"other node"<<otherNode<<" edge "<<edge<<"\n";
                        if(!mask.useEdge(edge) || !mask.useNode(otherNode)){
                            //std::cout<<"cont..\n";
                            return ContinueSeach;
                        }
                        const auto p = parents_[otherNode];
                        if(parents_[otherNode] < 0 && q == 0){
                            //std::cout<<"aaaa\n";
                            detail_bfs::singleSourceSingleTargetHelper(parents_, v, otherNode, path_);
                            //NIFTY_ASSERT(path_[0] == source,"");
                            //NIFTY_ASSERT(path_.back() == target,"");
                            return ReturnTrue;
                        }
                        else if(parents_[otherNode] > 0 && q == 1){
                            //std::cout<<"bbb\n";
                            detail_bfs::singleSourceSingleTargetHelper(parents_, otherNode, v, path_);
                            //NIFTY_ASSERT(path_[0] == source,"");
                            //NIFTY_ASSERT(path_.back() == target,"");
                            return ReturnTrue;
                        }
                        else if(parents_[otherNode] == 0){
                            ////std::cout<<"cccc\n";
                            if(q == 0){
                                //std::cout<<"cccc q=0\n";
                                parents_[otherNode] = v + 1;
                            }
                            else{
                                //std::cout<<"cccc q=1\n";
                                parents_[otherNode] = -static_cast<int64_t>(v) - 1;
                            }
                            queues_[q].push(otherNode);
                        }
                        else{
                            return ContinueSeach;
                        }
                        return ContinueSeach;
                    };

                    if(q == 0){
                        for(auto adj : graph_.adjacencyOut(v)){
                            auto retFlag = fHelper(adj.node(), adj.edge());
                            if(retFlag == ContinueSeach){
                                //continue;
                            }
                            else if(retFlag == ReturnTrue)
                                return true;
                        }
                    }
                    else{
                        for(auto adj : graph_.adjacencyIn(v)){
                            auto retFlag = fHelper(adj.node(), adj.edge());
                            if(retFlag == ContinueSeach){
                                //continue;
                            }
                            else if(retFlag == ReturnTrue)
                                return true;
                        }          
                    }
                }
                if(queues_[0].empty() && queues_[1].empty()) {
                    return false;
                }
            }
            //NIFTY_ASSERT(false,"");
        }
        const DequeType & path() const {
            return path_;
        }
    private:


        const GraphType & graph_;
        Parents parents_;
        std::array< std::queue<uint64_t> , 2> queues_;
        DequeType path_;
    };
} // namespace nifty::graph
} // namespace nifty

