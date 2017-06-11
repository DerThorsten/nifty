#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_andres_greedy_additive.hxx"

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_solver.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace lifted_multicut{


    template<class OBJECTIVE>
    void exportLiftedMulticutAndresGreedyAdditiveT(py::module & liftedMulticutModule) {

        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutAndresGreedyAdditive<ObjectiveType> Solver;
        typedef typename Solver::SettingsType SettingsType;
        
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule,"LiftedMulticutAndresGreedyAdditive")
            .def(py::init<>())
        ;
     
    }

    void exportLiftedMulticutAndresGreedyAdditive(py::module & liftedMulticutModule) {
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutAndresGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedMulticutAndresGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        }
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutAndresGreedyAdditiveT<ObjectiveType>(liftedMulticutModule);
        //}
    }

}
} // namespace nifty::graph::optimization
}
}
