#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../converter.hxx"
#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/gala/gala.hxx"



namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{



    using namespace py;
    //

    void exportGalaMainClass(py::module & galaModule) {

        typedef UndirectedGraph<> GraphType;
        typedef double FeatureValueType;
        typedef Gala<GraphType, FeatureValueType> GalaType;
        typedef typename GalaType::InstanceType InstanceType;
        typedef typename GalaType::TrainingInstanceType TrainingInstanceType;
        typedef typename GalaType::FeatureBaseTypeSharedPtr FeatureBaseTypeSharedPtr;
        typedef typename TrainingInstanceType::EdgeGtType EdgeGtType;

        py::class_<GalaType>(galaModule,"GalaUndirectedGraph")
            .def(py::init<>())
            .def("addInstance",[](
                GalaType * self, 
                TrainingInstanceType * trainingInstance
            ){
                self->addTrainingInstance(trainingInstance);
            },
                py::arg("trainingInstance"), 
                py::keep_alive<1,2>()
            )
        ;

        auto instanceCls = py::class_<InstanceType>(galaModule,"GalaInstanceUndirectedGraph");
        instanceCls
        ;

        auto trainingInstance = py::class_<TrainingInstanceType>(galaModule, "GalaTrainingInstanceUndirectedGraph", instanceCls);
        trainingInstance
        ;

        galaModule.def("galaInstance",
            [](
                const GraphType & graph,
                FeatureBaseTypeSharedPtr features
            ){
                auto ptr = new InstanceType(graph, features);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>(),
            py::arg("graph"), 
            py::arg("features")
        );

        galaModule.def("galaTrainingInstance",
            [](
                const GraphType & graph,
                FeatureBaseTypeSharedPtr features,
                nifty::marray::PyView<uint8_t, 1> edgeGt
            ){
                EdgeGtType egt(graph);
                for(const auto edge: graph.edges())
                    egt[edge] = edgeGt(edge);
                auto ptr = new TrainingInstanceType(graph, features, egt);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>(),
            py::arg("graph"), 
            py::arg("features"), 
            py::arg("edgeGt")
        );
    }

} // end namespace graph
} // end namespace nifty
    
