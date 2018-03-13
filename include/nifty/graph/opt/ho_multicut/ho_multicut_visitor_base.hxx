#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/visitor_base.hxx"


namespace nifty {
namespace graph {
namespace opt{
namespace ho_multicut{

    template<class OBJECTIVE>
    class HoMulticutBase;

    template<class OBJECTIVE>
    using HoMulticutVisitorBase = nifty::graph::opt::common::VisitorBase< HoMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using HoMulticutVerboseVisitor = nifty::graph::opt::common::VerboseVisitor< HoMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using HoMulticutEmptyVisitor = nifty::graph::opt::common::EmptyVisitor< HoMulticutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using HoMulticutVisitorProxy = nifty::graph::opt::common::VisitorProxy< HoMulticutBase<OBJECTIVE> >;

} // namespace nifty::graph::opt::ho_multicut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

