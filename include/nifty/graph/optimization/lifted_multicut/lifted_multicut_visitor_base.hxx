#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_VISITOR_BASE_HXX
#define NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_VISITOR_BASE_HXX

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/optimization/visitor_base.hxx"


namespace nifty {
namespace graph {
namespace lifted_multicut{


    template<class OBJECTIVE>
    class LiftedMulticutBase;

    template<class OBJECTIVE>
    using LiftedMulticutVisitorBase = graph::optimization::VisitorBase< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutVerboseVisitor = graph::optimization::VerboseVisitor< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutEmptyVisitor = graph::optimization::EmptyVisitor< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutVisitorProxy = graph::optimization::VisitorProxy< LiftedMulticutBase<OBJECTIVE> >;




} // namespace lifted_multicut
} // namespace graph
} // namespace nifty

#endif // #ifndef NIFTY_GRAPH_OPTIMIZATION_LIFTED_MULTICUT_LIFTED_MULTICUT_VISITOR_BASE_HXX
