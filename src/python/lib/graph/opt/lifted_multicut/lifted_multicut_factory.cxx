#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/opt/lifted_multicut/lifted_multicut_objective_name.hxx"


#include "nifty/python/converter.hxx"


#include "nifty/graph/opt/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/python/graph/opt/common/export_solver_factory.hxx"




namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{



    void exportLiftedMulticutFactory(py::module & module) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            typedef LiftedMulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                module, 
                LiftedMulticutObjectiveName<ObjectiveType>::name()
            );
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            typedef LiftedMulticutBase<ObjectiveType> SolverBaseType;
            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                module, 
                LiftedMulticutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::opt::lifted_multicut
} // namespace nifty::graph::opt
}
}
