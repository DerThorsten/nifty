#pragma once


#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace minstcut{

    template<class OBJECTIVE>
    class MinstcutBase;

    template<class OBJECTIVE>
    using MinstcutVisitorBase = nifty::graph::opt::common::VisitorBase< MinstcutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MinstcutVerboseVisitor = nifty::graph::opt::common::VerboseVisitor< MinstcutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MinstcutEmptyVisitor = nifty::graph::opt::common::EmptyVisitor< MinstcutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MinstcutVisitorProxy = nifty::graph::opt::common::VisitorProxy< MinstcutBase<OBJECTIVE> >;

} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

