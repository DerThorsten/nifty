#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace ufd{
    void exportUfd(py::module &);
}
}

    
PYBIND11_PLUGIN(_ufd) {

    py::options options;
    options.disable_function_signatures();
    
    py::module ufdModule("_ufd","ufd submodule");
    
    using namespace nifty::ufd;

    exportUfd(ufdModule);

    return ufdModule.ptr();
}
