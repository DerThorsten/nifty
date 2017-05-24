#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_graph_features.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace lifted_multicut{



    template<class OBJECTIVE>
    void exportLiftedGraphFeaturesT(py::module & liftedMulticutModule) {
        typedef OBJECTIVE ObjectiveType;


        liftedMulticutModule.def("liftedUcmFeatures",
            [](
                const ObjectiveType & objective,
                marray::PyView<double,1> edgeIndicators,
                marray::PyView<double,1> edgeSizes,
                marray::PyView<double,1> nodeSizes,
                std::vector<double > sizeRegularizers
            ){
                const size_t numberOfFeatures = sizeRegularizers.size() * 2;
                const size_t numberOfLiftedEdges = objective.numberOfLiftedEdges();
                marray::PyView<double> out({numberOfFeatures, numberOfLiftedEdges});

                {
                    py::gil_scoped_release allowThreads;
                    liftedUcmFeatures(objective, edgeIndicators,
                                      edgeSizes, nodeSizes,
                                      sizeRegularizers, out);
                }

                return out;

            },
            py::arg("objective"),
            py::arg("edgeIndicators"),
            py::arg("edgeSizes"),
            py::arg("nodeSizes"),
            py::arg("sizeRegularizers")
        );
       
    }
    
    void exportLiftedGraphFeatures(py::module & liftedMulticutModule){
        {
            typedef PyUndirectedGraph GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedGraphFeaturesT<ObjectiveType>(liftedMulticutModule);
        }
        {
            typedef nifty::graph::UndirectedGridGraph<2,true> GraphType;
            typedef LiftedMulticutObjective<GraphType, double> ObjectiveType;
            exportLiftedGraphFeaturesT<ObjectiveType>(liftedMulticutModule);
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
