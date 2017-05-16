#pragma once


#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/optimization/common/visitor_base.hxx"

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class MincutBase;

    template<class OBJECTIVE>
    using MincutVisitorBase = nifty::graph::optimization::common::VisitorBase< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutVerboseVisitor = nifty::graph::optimization::common::VerboseVisitor< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutEmptyVisitor = nifty::graph::optimization::common::EmptyVisitor< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutVisitorProxy = nifty::graph::optimization::common::VisitorProxy< MincutBase<OBJECTIVE> >;


} // namespace graph
} // namespace nifty

