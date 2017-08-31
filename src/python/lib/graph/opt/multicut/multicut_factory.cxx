#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective.hxx"
#include "nifty/python/graph/opt/multicut/multicut_objective_name.hxx"
#include "nifty/python/graph/opt/common/export_solver_factory.hxx"


#include "nifty/graph/opt/multicut/multicut_base.hxx"

#include "nifty/python/converter.hxx"





namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace opt{
namespace multicut{



    void exportMulticutFactory(py::module & multicutModule) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            typedef MulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                multicutModule, 
                MulticutObjectiveName<ObjectiveType>::name()
            );
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            typedef MulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                multicutModule, 
                MulticutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
