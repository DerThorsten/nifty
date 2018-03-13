#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/solver_base.hxx"
#include "nifty/graph/opt/multicut/multicut_visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace multicut{


    template<class OBJECTIVE>
    class MulticutBase :
        public nifty::graph::opt::common::SolverBase<
            OBJECTIVE,
            MulticutBase<OBJECTIVE>
        >
    {

    };

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

