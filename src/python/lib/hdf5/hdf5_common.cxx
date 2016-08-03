#ifdef WITH_HDF5
#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/marray/marray_hdf5.hxx"

namespace py = pybind11;


namespace nifty{
namespace hdf5{

    using namespace marray::hdf5;


    void exportHdf5Common(py::module & hdf5Module) {


        py::enum_<FileAccessMode>(hdf5Module, "FileAccessMode")
            .value("READ_ONLY", FileAccessMode::READ_ONLY)
            .value("READ_WRITE", FileAccessMode::READ_WRITE)
            //.export_values();
        ;

        py::enum_<HDF5Version>(hdf5Module, "HDF5Version")
            .value("DEFAULT_HDF5_VERSION", HDF5Version::DEFAULT_HDF5_VERSION)
            .value("LATEST_HDF5_VERSION", HDF5Version::LATEST_HDF5_VERSION)
            //.export_values();
        ;

        py::class_<hid_t>(hdf5Module, "HidT")
        ;


        hdf5Module.def("createFile", &createFile,
            py::arg("filename"),
            py::arg_t<HDF5Version>("hdf5version",LATEST_HDF5_VERSION)
        );

        hdf5Module.def("openFile", &openFile,
            py::arg("filename"),
            py::arg_t<FileAccessMode>("hdf5version",READ_ONLY),
            py::arg_t<HDF5Version>("hdf5version",LATEST_HDF5_VERSION)
        );

        hdf5Module.def("closeFile", &closeFile,
            py::arg("hidT")
        );

        hdf5Module.def("createGroup", &createGroup,
            py::arg("hidT"),
            py::arg("groupName")
        );

        hdf5Module.def("openGroup", &openGroup,
            py::arg("hidT"),
            py::arg("groupName")
        );

        hdf5Module.def("closeGroup", &closeGroup,
            py::arg("hidT")
        );
    }

}
}


#endif
