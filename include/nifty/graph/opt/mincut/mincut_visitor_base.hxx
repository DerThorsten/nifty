#pragma once


#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace mincut{

    template<class OBJECTIVE>
    class MincutBase;

    template<class OBJECTIVE>
    using MincutVisitorBase = nifty::graph::opt::common::VisitorBase< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutVerboseVisitor = nifty::graph::opt::common::VerboseVisitor< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutEmptyVisitor = nifty::graph::opt::common::EmptyVisitor< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutVisitorProxy = nifty::graph::opt::common::VisitorProxy< MincutBase<OBJECTIVE> >;

} // namespace nifty::graph::opt::mincut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

