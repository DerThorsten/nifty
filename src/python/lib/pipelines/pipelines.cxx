#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
namespace pipelines{



    


}
}


PYBIND11_PLUGIN(_pipelines) {
    py::module pipelinesModule("_pipelines", "pipelines submodule of nifty");

    using namespace nifty::pipelines;

        
        
    return pipelinesModule.ptr();
}

