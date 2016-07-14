#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
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
        typedef typename GalaType::Settings GalaSettingsType;
        typedef typename GalaType::InstanceType InstanceType;
        typedef typename GalaType::TrainingInstanceType TrainingInstanceType;
        typedef typename TrainingInstanceType::FeatureBaseType FeatureBaseType;



        py::class_<GalaSettingsType>(galaModule,"GalaSettingsUndirectedGraph")
            .def(py::init<>())
            .def_readwrite("threshold0", &GalaSettingsType::threshold0)
            .def_readwrite("threshold1", &GalaSettingsType::threshold1)
            .def_readwrite("thresholdU", &GalaSettingsType::thresholdU)
            .def_readwrite("numberOfEpochs", &GalaSettingsType::numberOfEpochs)
            .def_readwrite("numberOfTrees", &GalaSettingsType::numberOfTrees)
            .def_readwrite("mapFactory", &GalaSettingsType::mapFactory)
            .def_readwrite("perturbAndMapFactory", &GalaSettingsType::perturbAndMapFactory)
        ;

        py::class_<GalaType>(galaModule,"GalaUndirectedGraph")
            .def(py::init<const GalaSettingsType & >(),py::arg("settings"))
            .def(py::init<>())
            .def("addTrainingInstance",[](
                GalaType * self, 
                TrainingInstanceType & trainingInstance
            ){
                self->addTrainingInstance(trainingInstance);
            },
                py::arg("trainingInstance"), 
                py::keep_alive<1,2>()
            )

            .def("train",[](GalaType & gala){
                gala.train();
            })


            .def("predict",[](
                GalaType & gala, 
                InstanceType & instance

            ){
                nifty::marray::PyView<uint64_t, 1> nodeLabels({instance.graph().numberOfNodes()});
                gala.predict(instance, nodeLabels);
                return nodeLabels;
            })
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
                FeatureBaseType * features
            ){
                auto ptr = new InstanceType(graph, features);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>(),
            py::keep_alive<0,2>(),
            py::arg("graph"), 
            py::arg("features")
        );

        galaModule.def("galaTrainingInstance",
            [](
                const GraphType & graph,
                FeatureBaseType * features,
                nifty::marray::PyView<double, 1> edgeGt
            ){
                auto ptr = new TrainingInstanceType(graph, features, edgeGt);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>(),
            py::keep_alive<0,2>(),
            py::arg("graph"), 
            py::arg("features"), 
            py::arg("edgeGt")
        );

        galaModule.def("galaTrainingInstance",
            [](
                const GraphType & graph,
                FeatureBaseType * features,
                nifty::marray::PyView<double, 1> edgeGt,
                nifty::marray::PyView<double, 1> edgeGtUncertainty
            ){
                auto ptr = new TrainingInstanceType(graph, features, edgeGt, edgeGtUncertainty);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0,1>(),
            py::keep_alive<0,2>(),
            py::arg("graph"), 
            py::arg("features"), 
            py::arg("edgeGt"),
            py::arg("edgeGtUncertainty")
        );
    }

} // end namespace graph
} // end namespace nifty
    
