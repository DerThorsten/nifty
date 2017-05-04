#ifdef WITH_LP_MP

#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_mp.hxx"

#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

// FIXME only for hacky rounder, remove once this is done properly
#include "nifty/graph/optimization/multicut/multicut_greedy_additive.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{

    // -> ask thorstem
    typedef LP_MP::KlRounder DefaultRounder;

    // TODO this should just accept a mulituct factory
    // then we can choose any nifty solver at runtime
    // for now hard-code to greedy additive (doesn't need complicated params...)
    // TODO api for changing the objective / initialize objective at construction time
    struct NiftyRounder {
        
        typedef nifty::graph::UndirectedGraph<> GraphType;
        typedef MulticutObjective<GraphType, double> Objective;
        typedef MulticutGreedyAdditive<Objective> Solver;
        typedef typename Solver::NodeLabels NodeLabels;

        NiftyRounder() {}

        // TODO do we have to call by value here due to using async or could we also use a call by refernce?
        // TODO need to change between between edge and node labelings -> could be done more efficient ?!
        std::vector<char> operator()(GraphType g, std::vector<double> edgeValues) {

            std::vector<char> labeling(g.numberOfEdges(), 0);
            if(g.numberOfEdges() > 0) {
                
                Objective obj(g);
                auto & objWeights = obj.weights();
                for(auto eId = 0; eId < edgeValues.size(); ++eId) {
                    objWeights[eId] = edgeValues[eId];
                }
               
                Solver solver(obj);
                NodeLabels nodeLabeling(g.numberOfNodes());
                solver.optimize(nodeLabeling, nullptr);
                // node labeling to edge labeling
                for(auto eId = 0; eId < g.numberOfEdges(); ++eId) {
                    auto uv = g.uv(eId);
                    labeling[eId] = uv.first != uv.second;
                }

            }
            return labeling;

        }

        static std::string name() {
            return "NiftyRounder";
        }
    };

    template<class OBJECTIVE, class ROUNDER>
    void exportMulticutMpT(py::module & multicutModule){
        
        typedef OBJECTIVE ObjectiveType;
        typedef MulticutMp<ObjectiveType, ROUNDER> Solver;
        typedef typename Solver::Settings Settings;
        
        const auto solverName = std::string("MulticutMp") + ROUNDER::name();
        // FIXME nIter and verbose have no effect yet
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str())
            .def(py::init<>())
            .def_readwrite("verbose",&Settings::verbose)
            .def_readwrite("numberOfIterations",&Settings::numberOfIterations)
            .def_readwrite("primalComputationInterval",&Settings::primalComputationInterval)
            .def_readwrite("standardReparametrization",&Settings::standardReparametrization)
            .def_readwrite("roundingReparametrization",&Settings::roundingReparametrization)
            .def_readwrite("tightenReparametrization",&Settings::tightenReparametrization)
            .def_readwrite("tighten",&Settings::tighten)
            .def_readwrite("tightenInterval",&Settings::tightenInterval)
            .def_readwrite("tightenIteration",&Settings::tightenIteration)
            .def_readwrite("tightenSlope",&Settings::tightenSlope)
            .def_readwrite("tightenConstraintsPercentage",&Settings::tightenConstraintsPercentage)
            .def_readwrite("numberOfIterations",&Settings::numberOfIterations)
            .def_readwrite("minDualImprovement",&Settings::minDualImprovement)
            .def_readwrite("minDualImprovementInterval",&Settings::minDualImprovementInterval)
            .def_readwrite("timeout",&Settings::timeout)
        ; 

    }

    
    void exportMulticutMp(py::module & multicutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutMpT<ObjectiveType,DefaultRounder>(multicutModule);
            exportMulticutMpT<ObjectiveType,NiftyRounder>(multicutModule);
        }
        {
            typedef PyContractionGraph<PyUndirectedGraph> GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutMpT<ObjectiveType,DefaultRounder>(multicutModule);
            exportMulticutMpT<ObjectiveType,NiftyRounder>(multicutModule);
        }     

    }

} // namespace graph
} // namespace nifty
#endif
