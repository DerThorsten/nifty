#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/graph/optimization/multicut/multicut_objective_name.hxx"
#include "nifty/python/graph/optimization/common/export_solver_factory.hxx"


#include "nifty/graph/optimization/multicut/multicut_base.hxx"

#include "nifty/python/converter.hxx"





namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{



    void exportMulticutFactory(py::module & multicutModule) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            typedef MulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::optimization::common::exportSolverFactory<SolverBaseType>(
                multicutModule, 
                MulticutObjectiveName<ObjectiveType>::name()
            );
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            typedef MulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::optimization::common::exportSolverFactory<SolverBaseType>(
                multicutModule, 
                MulticutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
}
}
