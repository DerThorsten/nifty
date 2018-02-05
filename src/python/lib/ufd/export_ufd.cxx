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
               py::arg("numberOfIndices"),
                "This function does bla bla bla. \n\n"
                "Detailed....TODO"
            )
            // find for a single element
            .def("find", [](UfdType & self, const T index) {
                return self.find(index);
            },
                py::arg("element2"),
                "Find the representative element of the set that contains the given element (with path compression).\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   element (int): Element.\n\n"
            )
            .def("find", [](const UfdType & self, const T index) {
                return self.find(index);
            },
                py::arg("element2"),
                "Find the representative element of the set that contains the given element (with path compression).\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   element (int): Element.\n\n"
            )
            // find vectorized
            .def("find", [](UfdType & self, const marray::PyView<T,1> indices) {
                marray::PyView<IndexType,1> out({indices.shape(0)});
                for(int i = 0; i < indices.shape(0); ++i)
                    out(i) = self.find(indices(i));
                return out;
            },
                py::arg("element2"),
                "Find the representative element of the set that contains the given element (with path compression).\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   element (int): Element.\n\n"
            )
            .def("find", [](const UfdType & self, const marray::PyView<T,1> indices) {
                marray::PyView<IndexType,1> out({indices.shape(0)});
                for(int i = 0; i < indices.shape(0); ++i)
                    out(i) = self.find(indices(i));
                return out;
            },
                py::arg("element2"),
                "Find the representative element of the set that contains the given element (with path compression).\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   element (int): Element.\n\n"
            )
            // merge for a single element
            .def("merge", &UfdType::merge,
                py::arg("element1"),
                py::arg("element2"),
                " Merge two elements\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   element1 (int): Element in the first set.\n"
                "   element2 (int): Element in the first set.\n\n"
                ) 
            // merge vectorized
            .def("merge", [](UfdType & self, marray::PyView<T,2> mergeIndices) {
                NIFTY_CHECK_OP(mergeIndices.shape(1),==,2,"We need pairs of indices for merging!")
                for(int i = 0; i < mergeIndices.shape(0); ++i)
                    self.merge(mergeIndices(i,0), mergeIndices(i,1));
            },
                " Merge two elements\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
            )
            .def("assign", &UfdType::assign,
                py::arg("size"),
                "Reset the ufd (to a number of sets each containing one element).\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   size (int): Number of distinct sets.\n\n"
                )
            .def("reset", &UfdType::reset)
            .def("insert", &UfdType::insert,
                py::arg("number"),
                "Insert a number of new sets, each containing one element.\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
                "   number (int): Number of sets to insert.\n\n"
                )
            .def_property_readonly("numberOfElements", &UfdType::numberOfElements,
                "returns the number of elements.\n\n"
                "Detailed....TODO\n\n"
                )
            .def_property_readonly("numberOfSets", &UfdType::numberOfSets,
                "returns the number of sets.\n\n"
                "Detailed....TODO\n\n"
                )
            .def("elementLabeling", [](const UfdType & self) {
                marray::PyView<IndexType,1> out({self.numberOfElements()});
                self.elementLabeling(&out(0));
                return out;
            },
                "Output a contiguous labeling of all elements.\n\n"
                "Detailed....TODO\n\n"
            )
            .def("representativesToSets", [](const UfdType & self) {
                std::vector<std::vector<IndexType>> out;
                self.representativesToSets(out);
                return out;
            },
                "returns the number of elements.\n\n"
                "Detailed....TODO\n\n"
            )
        ;
    }

    void exportUfd(py::module & ufdModule) {
        exportUfdT<uint32_t>(ufdModule, "Ufd_UInt32");
        exportUfdT<uint64_t>(ufdModule, "Ufd_UInt64");
    }

}
}
