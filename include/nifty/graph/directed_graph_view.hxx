#pragma once

#include "nifty/graph/directed_graph_base.hxx"

namespace nifty{
namespace graph{

    template<class GRAPH>
    class DirectedGraphView{
    public:
        typedef GRAPH GraphType;
    private:    
        const GraphType & graph_;
    };

} // namespace nifty::graph
} // namespace nifty

