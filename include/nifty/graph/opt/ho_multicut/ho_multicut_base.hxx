#pragma once

#include <string>
#include <initializer_list>
#include <sstream>

#include "nifty/graph/opt/common/solver_base.hxx"
#include "nifty/graph/opt/ho_multicut/ho_multicut_visitor_base.hxx"

namespace nifty {
namespace graph {
namespace opt{
namespace ho_multicut{


    template<class OBJECTIVE>
    class HoMulticutBase :
        public nifty::graph::opt::common::SolverBase<
            OBJECTIVE,
            HoMulticutBase<OBJECTIVE>
        >
    {

    };

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
} // namespace graph
} // namespace nifty

