#pragma once

#include "nifty/graph/subgraph_mask.hxx"
#include "nifty/graph/detail/search_impl.hxx"

namespace nifty{
namespace graph{

    template<class GRAPH>
    using BreadthFirstSearch = detail_graph::SearchImpl<GRAPH, detail_graph::FiFo<int64_t> >;

} // namespace nifty::graph
} // namespace nifty

