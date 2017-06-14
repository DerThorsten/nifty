#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/visitor_base.hxx"


namespace nifty {
namespace graph {
namespace opt{
namespace multicut{

    template<class OBJECTIVE>
    class MulticutBase;

    template<class OBJECTIVE>
    using MulticutVisitorBase = nifty::graph::opt::common::VisitorBase< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutVerboseVisitor = nifty::graph::opt::common::VerboseVisitor< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutEmptyVisitor = nifty::graph::opt::common::EmptyVisitor< MulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MulticutVisitorProxy = nifty::graph::opt::common::VisitorProxy< MulticutBase<OBJECTIVE> >;

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

