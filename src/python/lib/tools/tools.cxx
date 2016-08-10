#ifdef WITH_HDF5

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;




namespace nifty{
namespace tools{


    void exportMakeDense(py::module &);


    void initSubmoduleTools(py::module &niftyModule) {
        auto toolsModule = niftyModule.def_submodule("tools","tools submodule");
        exportMakeDense(toolsModule);

    }

}
}

#endif
