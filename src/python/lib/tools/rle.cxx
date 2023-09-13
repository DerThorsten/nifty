#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <typeinfo> // to debug atm

#include "xtensor-python/pyarray.hpp"
#include "nifty/tools/rle.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T>
    void exportComputeRLET(py::module & toolsModule) {

        toolsModule.def("computeRLE",
        [](xt::pyarray<T> & dataIn){

            std::vector<int> counts;
            {
                py::gil_scoped_release allowThreads;
                tools::computeRLE(dataIn, counts);
            }
            return counts;
        });
    }


    void exportComputeRLE(py::module & toolsModule) {
        exportComputeRLET<bool>(toolsModule);
    }

}
}
