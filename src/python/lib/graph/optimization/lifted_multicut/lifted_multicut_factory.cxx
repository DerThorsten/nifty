#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective_name.hxx"


#include "nifty/python/converter.hxx"


#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/python/graph/optimization/common/export_solver_factory.hxx"




namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{



    void exportLiftedMulticutFactory(py::module & module) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            typedef LiftedMulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::optimization::common::exportSolverFactory<SolverBaseType>(
                module, 
                LiftedMulticutObjectiveName<ObjectiveType>::name()
            );
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            typedef LiftedMulticutBase<ObjectiveType> SolverBaseType;
            nifty::graph::optimization::common::exportSolverFactory<SolverBaseType>(
                module, 
                LiftedMulticutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::optimization::lifted_multicut
} // namespace nifty::graph::optimization
}
}
