#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace lifted_multicut{


    template<class OBJECTIVE>
    class LiftedMulticutBase;

    template<class OBJECTIVE>
    using LiftedMulticutVisitorBase =  nifty::graph::opt::common::VisitorBase< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutVerboseVisitor =  nifty::graph::opt::common::VerboseVisitor< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutEmptyVisitor =  nifty::graph::opt::common::EmptyVisitor< LiftedMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using LiftedMulticutVisitorProxy =  nifty::graph::opt::common::VisitorProxy< LiftedMulticutBase<OBJECTIVE> >;




} // namespace lifted_multicut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

