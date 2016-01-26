#pragma once
#ifndef NIFTY_GRAPH_DIGRAPH_VIEW_HXX
#define NIFTY_GRAPH_DIGRAPH_VIEW_HXX

#include "nifty/graph/directed_graph_base.hxx"

namespace nifty{
namespace graph{

    template<class GRAPH>
    class DirectedGraphView{
    public:
        typedef GRAPH Graph;
    private:    
        const Graph & graph_;
    };

} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_DIGRAPH_VIEW_HXX
