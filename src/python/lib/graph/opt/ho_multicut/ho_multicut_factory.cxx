#include <pybind11/pybind11.h>


#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective.hxx"
#include "nifty/python/graph/opt/ho_multicut/ho_multicut_objective_name.hxx"
#include "nifty/python/graph/opt/common/export_solver_factory.hxx"


#include "nifty/graph/opt/ho_multicut/ho_multicut_base.hxx"

#include "nifty/python/converter.hxx"





namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(McBase, std::shared_ptr<McBase>);

namespace nifty{
namespace graph{
namespace opt{
namespace ho_multicut{



    void exportHoMulticutFactory(py::module & hoMulticutModule) {


        {
            typedef PyUndirectedGraph GraphType;
            typedef HoMulticutObjective<GraphType, double> ObjectiveType;
            typedef HoMulticutBase<ObjectiveType> SolverBaseType;

            nifty::graph::opt::common::exportSolverFactory<SolverBaseType>(
                hoMulticutModule, 
                HoMulticutObjectiveName<ObjectiveType>::name()
            );
        }
    }

} // namespace nifty::graph::opt::multicut
} // namespace nifty::graph::opt
}
}
