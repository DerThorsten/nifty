#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/minstcut/minstcut_objective.hxx"


#include "nifty/python/converter.hxx"


#include "nifty/graph/opt/minstcut/minstcut_base.hxx"
#include "nifty/python/graph/opt/common/export_solver_factory.hxx"




namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace opt{
namespace minstcut{



    void exportMinstcutFactory(py::module & multicutModule) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            typedef MinstcutBase<ObjectiveType> SolverBaseType;

            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                multicutModule,
                MinstcutObjectiveName<ObjectiveType>::name()
            );
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MinstcutObjective<GraphType, double> ObjectiveType;
            typedef MinstcutBase<ObjectiveType> SolverBaseType;

            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                multicutModule,
                MinstcutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::opt::minstcut
} // namespace nifty::graph::opt
}
}
