#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

namespace nifty{
namespace malis{
    void exportMalisGradient(py::module &);
}
}


PYBIND11_PLUGIN(_malis) {
    py::module malisModule("_malis", "malis submodule of nifty");

    using namespace nifty::malis;

    exportMalisGradient(malisModule);
        
    return malisModule.ptr();
}

