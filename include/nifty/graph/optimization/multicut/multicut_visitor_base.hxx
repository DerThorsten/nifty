#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/optimization/common/visitor_base.hxx"


namespace nifty {
namespace graph {
namespace optimization{
namespace multicut{

    template<class OBJECTIVE>
    class MulticutBase;

    template<class OBJECTIVE>
    using MulticutVisitorBase = nifty::graph::optimization::common::VisitorBase< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutVerboseVisitor = nifty::graph::optimization::common::VerboseVisitor< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutEmptyVisitor = nifty::graph::optimization::common::EmptyVisitor< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutVisitorProxy = nifty::graph::optimization::common::VisitorProxy< MulticutBase<OBJECTIVE> >;

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty

