#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "nifty/python/converter.hxx"
#include "nifty/ufd/ufd.hxx"

namespace py = pybind11;


namespace nifty{
namespace ufd{


    template<class T>
    void exportUfdT(py::module & ufdModule) {
    
        typedef Ufd<T> UfdType;
        typedef typename UfdType::Index IndexType;

        py::class_<UfdType>(ufdModule, "Ufd")
            .def(py::init<const IndexType>(),
               py::arg_t<IndexType>("numberOfIndices",0)
            )
            .def("find", [](UfdType & self, const T index) {
                return self.find(index);
            })
            .def("find", [](const UfdType & self, const T index) {
                return self.find(index);
            })
            .def("merge", &UfdType::merge)
            .def("assign", &UfdType::assign)
            .def("reset", &UfdType::reset)
            .def("insert", &UfdType::insert)
            .def_property_readonly("numberOfElements", &UfdType::numberOfElements)
            .def_property_readonly("numberOfSets", &UfdType::numberOfSets)
            .def("elementLabeling", [](const UfdType & self) {
                marray::PyView<IndexType,1> out({self.numberOfElements()});
                self.elementLabeling(&out(0));
                return out;
            })
            .def("representativesToSets", [](const UfdType & self) {
                std::vector<std::vector<IndexType>> out;
                self.representativesToSets(out);
                return out;
            })
        ;
    }
    
    void initSubmoduleUfd(py::module &niftyModule) {
        auto ufdModule = niftyModule.def_submodule("ufd","ufd submodule");
        
        exportUfdT<uint32_t>(ufdModule);
        exportUfdT<uint64_t>(ufdModule);
    }

}
}
