#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/optimization/common/visitor_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{


    template<class OBJECTIVE>
    class LiftedMulticutBase;

    template<class OBJECTIVE>
    using LiftedMulticutVisitorBase =  nifty::graph::optimization::common::VisitorBase< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutVerboseVisitor =  nifty::graph::optimization::common::VerboseVisitor< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutEmptyVisitor =  nifty::graph::optimization::common::EmptyVisitor< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutVisitorProxy =  nifty::graph::optimization::common::VisitorProxy< LiftedMulticutBase<OBJECTIVE> >;




} // namespace lifted_multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

