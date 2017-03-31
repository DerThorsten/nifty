#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

#include "nifty/python/converter.hxx"
#include "nifty/ufd/ufd.hxx"

namespace nifty{
namespace ufd{


    template<class T>
    void exportUfdT(py::module & ufdModule, const std::string & clsName) {
    
        typedef Ufd<T> UfdType;
        typedef typename UfdType::Index IndexType;

        py::class_<UfdType>(ufdModule, clsName.c_str())
            .def(py::init<const IndexType>(),
               py::arg("numberOfIndices")
            )
            // find for a single element
            .def("find", [](UfdType & self, const T index) {
                return self.find(index);
            })
            .def("find", [](const UfdType & self, const T index) {
                return self.find(index);
            })
            // find vectorized
            .def("find", [](UfdType & self, const marray::PyView<T,1> indices) {
                marray::PyView<IndexType,1> out({indices.shape(0)});
                for(int i = 0; i < indices.shape(0); ++i)
                    out(i) = self.find(indices(i));
                return out;
            })
            .def("find", [](const UfdType & self, const marray::PyView<T,1> indices) {
                marray::PyView<IndexType,1> out({indices.shape(0)});
                for(int i = 0; i < indices.shape(0); ++i)
                    out(i) = self.find(indices(i));
                return out;
            })
            // merge for a single element
            .def("merge", &UfdType::merge) 
            // merge vectorized
            .def("merge", [](UfdType & self, marray::PyView<T,2> mergeIndices) {
                NIFTY_CHECK_OP(mergeIndices.shape(1),==,2,"We need pairs of indices for merging!")
                for(int i = 0; i < mergeIndices.shape(0); ++i)
                    self.merge(mergeIndices(i,0), mergeIndices(i,1));
            })
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

    void exportUfd(py::module & ufdModule) {
        exportUfdT<uint32_t>(ufdModule, "Ufd_UInt32");
        exportUfdT<uint64_t>(ufdModule, "Ufd_UInt64");
    }

}
}
