#include <pybind11/pybind11.h>


#include "nifty/python/graph/optimization/lifted_multicut/export_lifted_multicut_solver.hxx"
#include "nifty/python/converter.hxx"

#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_mp.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class OBJECTIVE>
    void exportLiftedMulticutMpT(py::module & liftedMulticutModule) {
        
        typedef OBJECTIVE ObjectiveType;
        typedef LiftedMulticutMp<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;
        typedef LiftedMulticutFactory<Solver> Factory;
        const auto solverName = std::string("LiftedMulticutMp");
        exportLiftedMulticutSolver<Solver>(liftedMulticutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("lmcFactory",&Settings::lmcFactory)
            .def_readwrite("greedyWarmstart",&Settings::greedyWarmstart)
        ;
    }
   

    void exportLiftedMulticutMp(py::module & liftedMulticutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType,double> ObjectiveType;
            exportLiftedMulticutMpT<ObjectiveType>(liftedMulticutModule);
        }

        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
        //    exportLiftedMulticutIlpT<ObjectiveType>(liftedMulticutModule);
        //}    
    }


}
}
}
