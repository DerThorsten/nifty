#ifdef WITH_HDF5
#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>

#include "nifty/python/converter.hxx"
#include "nifty/hdf5/hdf5.hxx"


namespace py = pybind11;


namespace nifty{
namespace hdf5{

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


        py::class_<CacheSettings>(hdf5Module, "CacheSettings")
            .def(py::init<int,int,float>(),
                py::arg("hashTabelSize") = 977,
                py::arg("nBytes") = 36000000,
                py::arg("rddc") = 1.0
            )
            .def_readwrite("hashTabelSize", &CacheSettings::hashTabelSize)
            .def_readwrite("nBytes", &CacheSettings::nBytes)
            .def_readwrite("rddc", &CacheSettings::rddc)
        ;



        hdf5Module.def("createFile", [](
            const std::string & filename,
            const CacheSettings & cacheSettings,
            const HDF5Version & hdf5version
        ){
            return createFile(filename, hdf5version, cacheSettings);
        },
            py::arg("filename"),
            py::arg("cacheSettings"),
            py::arg("hdf5version") = DEFAULT_HDF5_VERSION
        );

        hdf5Module.def("createFile", [](
            const std::string & filename,
            const HDF5Version & hdf5version
        ){
            return createFile(filename, hdf5version);
        },
            py::arg("filename"),
            py::arg("hdf5version") = DEFAULT_HDF5_VERSION
        );

        hdf5Module.def("openFile", [](
            const std::string & filename,
            const CacheSettings & cacheSettings,
            const FileAccessMode & fileAccessMode,
            const HDF5Version & hdf5version
        ){
            return openFile(filename, fileAccessMode, hdf5version, cacheSettings);
        },
            py::arg("filename"),
            py::arg("cacheSettings"),
            py::arg("fileAccessMode") = READ_ONLY,
            py::arg("hdf5version") = DEFAULT_HDF5_VERSION
        );


        hdf5Module.def("openFile", [](
            const std::string & filename,
            const FileAccessMode & fileAccessMode,
            const HDF5Version & hdf5version
        ){
            return openFile(filename, fileAccessMode, hdf5version);
        },
            py::arg("filename"),
            py::arg("fileAccessMode") = READ_ONLY,
            py::arg("hdf5version") = DEFAULT_HDF5_VERSION
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
