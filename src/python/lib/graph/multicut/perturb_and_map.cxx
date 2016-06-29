#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/graph/simple_graph.hxx"
#include "nifty/graph/multicut/multicut_objective.hxx"
#include "nifty/graph/multicut/perturb_and_map.hxx"
#include "../../converter.hxx"

namespace py = pybind11;


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);



namespace nifty{
namespace graph{



    void exportPerturbAndMap(py::module & multicutModule) {

        typedef UndirectedGraph<> Graph;
        typedef MulticutObjective<Graph, double> McObjective;
        typedef PerturbAndMap<McObjective> PerturbAndMapType;
        typedef typename PerturbAndMapType::Settings PerturbAndMapSettingsType;
        const auto clsName = std::string("PerturbAndMapUndirectedGraph");
        const auto settingsClsName = std::string("PerturbAndMapSettingsUndirectedGraph");
        auto multicutObjectiveCls = py::class_<PerturbAndMapType>(multicutModule, clsName.c_str())
        ;


        py::class_<PerturbAndMapSettingsType>(multicutModule, settingsClsName.c_str())
            .def(py::init<>())
            .def_readwrite("mcFactory",&PerturbAndMapSettingsType::mcFactory)
        ;



        multicutModule.def("perturbAndMap",
            [](const McObjective & objective, const PerturbAndMapSettingsType & s){
                auto perturbAndMap = new PerturbAndMapType(objective);
                return perturbAndMap;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("objective"),py::arg("settings")
        );

        
    }

}
}
