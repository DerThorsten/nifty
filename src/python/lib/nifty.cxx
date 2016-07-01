#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;



namespace nifty{
namespace graph{
    void initSubmoduleGraph(py::module & );
}
}

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_PLUGIN(_nifty) {
    py::module niftyModule("_nifty", "nifty python bindings");




    //y::implicitly_convertible<py::array_t<float>, nifty::NumpyArray<float> >();
    //py::implicitly_convertible<py::array_t<uint64_t>, nifty::NumpyArray<uint64_t> >();
    using namespace nifty;
    graph::initSubmoduleGraph(niftyModule);
}
