#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nifty/graph/agglo/cluster_policies/detail/merge_rules.hxx"


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{




    void exportMergeRules(py::module & aggloModule) {
        

        py::class_<merge_rules::ArithmeticMeanSettings>(aggloModule, "ArithmeticMeanSettings")
            .def(py::init<>())
            .def("__str__",[](const merge_rules::ArithmeticMeanSettings & self){
                return self.name();
            })
        ;

        py::class_<merge_rules::GeneralizedMeanSettings>(aggloModule, "GeneralizedMeanSettings")
            .def(py::init<double>(),
                py::arg("p")=1.0
            )
            .def("__str__",[](const merge_rules::GeneralizedMeanSettings & self){
                return self.name();
            })
        ;


        py::class_<merge_rules::SmoothMaxSettings>(aggloModule, "SmoothMaxSettings")
            .def(py::init<double>(),
                py::arg("p")=1.0
            )
            .def("__str__",[](const merge_rules::SmoothMaxSettings & self){
                return self.name();
            })
        ;

        py::class_<merge_rules::RankOrderSettings>(aggloModule, "RankOrderSettings")
            .def(py::init<double, uint16_t>(),
                py::arg("q")=0.5,
                py::arg("numberOfBins")=50
            )
            .def("__str__",[](const merge_rules::RankOrderSettings & self){
                return self.name();
            })
        ;

        py::class_<merge_rules::MaxSettings>(aggloModule, "MaxSettings")
            .def(py::init<>())
            .def("__str__",[](const merge_rules::MaxSettings & self){
                return self.name();
            })
        ;

        py::class_<merge_rules::MinSettings>(aggloModule, "MinSettings")
            .def(py::init<>())
            .def("__str__",[](const merge_rules::MinSettings & self){
                return self.name();
            })
        ;
    }

} // end namespace agglo
} // end namespace graph
} // end namespace nifty
    
