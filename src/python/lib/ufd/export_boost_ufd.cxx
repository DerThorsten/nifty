#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <xtensor-python/pytensor.hpp>

#include "nifty/ufd/boost_ufd.hxx"

namespace py = pybind11;


namespace nifty{
namespace ufd{


    template<class T>
    void exportBoostUfdT(py::module & ufdModule, const std::string & clsName) {

        typedef BoostUfd<T> UfdType;
        typedef typename UfdType::value_type IndexType;

        py::class_<UfdType>(ufdModule, clsName.c_str())
            .def(py::init<xt::pytensor<IndexType, 1>, std::size_t>(),
               py::arg("elements"), py::arg("upper_bound"),
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

            // find vectorized
            .def("find", [](UfdType & self, const xt::pytensor<T, 1> & indices) {
                const unsigned int n_indices = indices.shape()[0];
                xt::pytensor<IndexType, 1> out = xt::zeros<IndexType>({n_indices});
                for(int i = 0; i < n_indices; ++i) {
                    out(i) = self.find(indices(i));
                }
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
            .def("merge", [](UfdType & self, xt::pytensor<T, 2> & mergeIndices) {
                // NIFTY_CHECK_OP(mergeIndices.shape(1),==,2,"We need pairs of indices for merging!")
                for(int i = 0; i < mergeIndices.shape()[0]; ++i)
                    self.merge(mergeIndices(i, 0), mergeIndices(i, 1));
            },
                " Merge two elements\n\n"
                "Detailed....TODO\n\n"
                "Args:\n"
            )
        ;
    }

    void exportBoostUfd(py::module & ufdModule) {
        exportBoostUfdT<uint32_t>(ufdModule, "BoostUfd_UInt32");
        exportBoostUfdT<uint64_t>(ufdModule, "BoostUfd_UInt64");
    }

}
}
