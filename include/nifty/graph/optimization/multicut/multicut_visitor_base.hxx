#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_VISITOR_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_VISITOR_BASE_HXX

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/optimization/visitor_base.hxx"

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class MulticutBase;

    template<class OBJECTIVE>
    using MulticutVisitorBase = optimization::VisitorBase< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutVerboseVisitor = optimization::VerboseVisitor< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutEmptyVisitor = optimization::EmptyVisitor< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutVisitorProxy = optimization::VisitorProxy< MulticutBase<OBJECTIVE> >;


} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_MULTICUT_VISITOR_BASE_HXX
