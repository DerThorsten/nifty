#ifdef WITH_HDF5

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;




namespace nifty{
namespace hdf5{


    void exportHdf5Common(py::module &);
    void exportHdf5Array(py::module &);

    void initSubmoduleHdf5(py::module &niftyModule) {
        auto hdf5Module = niftyModule.def_submodule("hdf5","hdf5 submodule");
        exportHdf5Common(hdf5Module);
        exportHdf5Array(hdf5Module);
    }

}
}

#endif
