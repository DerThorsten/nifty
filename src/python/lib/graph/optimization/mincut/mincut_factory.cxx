#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/mincut/mincut_objective.hxx"


#include "nifty/python/converter.hxx"


#include "nifty/graph/optimization/mincut/mincut_base.hxx"
#include "nifty/python/graph/optimization/common/export_solver_factory.hxx"




namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{



    void exportMincutFactory(py::module & multicutModule) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            typedef MincutBase<ObjectiveType> SolverBaseType;

            nifty::graph::optimization::common::exportSolverFactory<SolverBaseType>(
                multicutModule, 
                MincutObjectiveName<ObjectiveType>::name()
            );
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MincutObjective<GraphType, double> ObjectiveType;
            typedef MincutBase<ObjectiveType> SolverBaseType;

            nifty::graph::optimization::common::exportSolverFactory<SolverBaseType>(
                multicutModule, 
                MincutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::optimization::mincut
} // namespace nifty::graph::optimization
}
}
