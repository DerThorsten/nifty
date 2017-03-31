#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nifty{
namespace deep_learning{
    //void exportUfd(py::module &);
}
}

    
PYBIND11_PLUGIN(_deep_learning) {
    py::module deep_learningModule("_deep_learning","deep_learning submodule");
    
    using namespace nifty::deep_learning;

    //exportUfd(deep_learningModule);

    return deep_learningModule.ptr();
}
