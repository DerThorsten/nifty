
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include <iostream>

#include "nifty/tools/blocking.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace pipelines{
namespace ilastik_backend{

    void exportInteractivePixelClassification(py::module & mod);

}
}
}




PYBIND11_PLUGIN(_ilastik_backend) {
    py::module mod("_ilastik_backend", "neuro seg submodule of nifty");

    using namespace nifty::pipelines::ilastik_backend;


    exportInteractivePixelClassification(mod);
        
    return mod.ptr();
}

