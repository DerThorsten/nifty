#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{

} // namespace nifty::graph::opt
}
}



PYBIND11_MODULE(_opt, module) {
    module.doc() =  "_opt", "opt submodule of nifty.graph";
    using namespace nifty::graph::opt;
}

