#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace ufd{
    void exportUfd(py::module &);
}
}

    
PYBIND11_MODULE(_ufd, ufdModule) {

    py::options options;
    options.disable_function_signatures();
    
    ufdModule.doc() = "ufd submodule";
    
    using namespace nifty::ufd;

    exportUfd(ufdModule);

}
