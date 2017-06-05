#pragma once


namespace nifty{
namespace graph{


    template<class G>
    struct DefaultSubgraphMask{
        bool useEdge(const uint64_t)const{
            return true;
        }
        bool useNode(const uint64_t)const{
            return true;
        }
    };


} // namespace nifty::graph
} // namespace nifty

