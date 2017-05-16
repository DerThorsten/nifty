#pragma once


#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/optimization/visitor_base.hxx"

namespace nifty {
namespace graph {


    template<class OBJECTIVE>
    class MincutBase;

    template<class OBJECTIVE>
    using MincutVisitorBase = optimization::VisitorBase< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutVerboseVisitor = optimization::VerboseVisitor< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutEmptyVisitor = optimization::EmptyVisitor< MincutBase<OBJECTIVE> >;

    template<class OBJECTIVE>
    using MincutVisitorProxy = optimization::VisitorProxy< MincutBase<OBJECTIVE> >;


} // namespace graph
} // namespace nifty

