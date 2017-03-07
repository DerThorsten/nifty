#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

namespace nifty{
namespace malis{
    void exportMalisLoss(py::module &);
}
}


PYBIND11_PLUGIN(_malis) {
    py::module malisModule("_malis", "malis submodule of nifty");

    using namespace nifty::malis;

    exportMalisLoss(malisModule);
        
    return malisModule.ptr();
}

