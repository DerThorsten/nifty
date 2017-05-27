#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_andres.hxx"

#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    template<class OBJECTIVE>
    void exportMulticutAndresT(py::module & multicutModule){
        
        typedef OBJECTIVE ObjectiveType;

        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        
        // export greedy additive
        {


            ///////////////////////////////////////////////////////////////
            // DOCSTRING HELPER
            ///////////////////////////////////////////////////////////////
            nifty::graph::optimization::SolverDocstringHelper docHelper;
            docHelper.objectiveName =
                "multicut objective";
            docHelper.objectiveClsName = 
                MulticutObjectiveName<OBJECTIVE>::name();
            docHelper.name = 
                "greedy additive andres";
            docHelper.mainText =  
                "Find approximate solutions via\n"
                "agglomerative clustering as in :cite:`beier_15_funsion`.\n";
            docHelper.cites.emplace_back("beier_15_funsion");
            docHelper.note = 
                "This solver should be used to\n"        
                "warm start other solvers with.\n"
                "This solver is very fast but\n"
                "yields rather suboptimal results.\n";
            docHelper.warning = 
                "This native nifty implementation as this implementation\n";
                "from andres which is just used for comparison\n";


            typedef MulticutAndresGreedyAdditive<ObjectiveType> Solver;
            typedef typename Solver::Settings Settings;
            const auto solverName = std::string("MulticutAndresGreedyAdditive");
            // FIXME verbose has no effect yet
            exportMulticutSolver<Solver>(multicutModule, solverName.c_str(), docHelper)
                .def(py::init<>())
            ; 
        }
        
        // export kernighan lin
        {



            ///////////////////////////////////////////////////////////////
            // DOCSTRING HELPER
            ///////////////////////////////////////////////////////////////
            nifty::graph::optimization::SolverDocstringHelper docHelper;
            docHelper.objectiveName = "multicut objective";
            docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
            docHelper.name = "Kernighan Lin";
            docHelper.mainText =  
            "KernighanLin Algorithm with joins for multicuts\n"
            "As introduced in TODO"; 
            docHelper.cites.emplace_back("TODO");
            docHelper.note = "This solver should be warm started,"
                            "otherwise  results are very poor."
                            "Using :func:`greedyAdditiveFactory` to create "
                            "a solver for warm starting is suggested.";




            typedef MulticutAndresKernighanLin<ObjectiveType> Solver;
            typedef typename Solver::Settings Settings;
            const auto solverName = std::string("MulticutAndresKernighanLin");
            // FIXME verbose has no effect yet
            exportMulticutSolver<Solver>(multicutModule, solverName.c_str(), docHelper)
                .def(py::init<>())
                .def_readwrite("numberOfInnerIterations", &Settings::numberOfInnerIterations)
                .def_readwrite("numberOfOuterIterations", &Settings::numberOfOuterIterations)
                .def_readwrite("epsilon", &Settings::epsilon)
                .def_readwrite("verbose", &Settings::verbose)
                .def_readwrite("greedyWarmstart", &Settings::greedyWarmstart)
            ; 
        }

    }

    
    void exportMulticutAndres(py::module & multicutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutAndresT<ObjectiveType>(multicutModule);
        }
        // FIXME this doesn't compile
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportMulticutMpT<ObjectiveType>(multicutModule);
        //}     

    }

} // namespace graph
} // namespace nifty
