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


        hdf5Module.def("setCacheOnFile", [](const hid_t & fileHandle){
            auto plist = H5Fget_access_plist(fileHandle);
            const auto anyVal = 0;
            const auto somePrime = 977;
            const auto nBytes = 36000000;
            const auto rddc = 1.0;
            auto ret = H5Pset_cache(plist, anyVal, somePrime,  nBytes, rddc);
            H5Pclose(plist);
            std::cout<<"set H5Pset_cache groupHandle_ "<<ret<<"\n";
        });

        hdf5Module.def("setCacheOnFile", [](const hid_t & fileHandle){
            auto plist = H5Fget_access_plist(fileHandle);
            const auto anyVal = 0;
            const auto somePrime = 977;
            const auto nBytes = 36000000;
            const auto rddc = 1.0;
            auto ret = H5Pset_cache(plist, anyVal, somePrime,  nBytes, rddc);
            H5Pclose(plist);
            std::cout<<"set H5Pset_cache groupHandle_ "<<ret<<"\n";
        });
        
        struct CacheReturnType {
            CacheReturnType(const double & somePrime, const double & nBytes, const double & rdcc)
                : somePrime_(somePrime), nBytes_(nBytes), rdcc_(rdcc)
            {}
            size_t getSomePrime() const {
                return somePrime_;
            }
            size_t getNBytes() const {
                return nBytes_;
            }
            double getRdcc() const {
                return rdcc_;
            }
        private:
            size_t somePrime_;
            size_t nBytes_;
            double rdcc_;
        };
        
        py::class_<CacheReturnType>(hdf5Module,"CacheReturnType")
            .def_property_readonly("somePrime", &CacheReturnType::getSomePrime)
            .def_property_readonly("nBytes", &CacheReturnType::getNBytes)
            .def_property_readonly("rdcc", &CacheReturnType::getRdcc)
        ;
        
        hdf5Module.def("getCacheOnFileImpl", [](const hid_t & fileHandle){
            auto plist = H5Fget_access_plist(fileHandle);
            int anyVal;
            size_t somePrime;
            size_t nBytes;
            double rdcc;
            auto ret = H5Pget_cache(plist, &anyVal, &somePrime, &nBytes, &rdcc);
            H5Pclose(plist);
            std::cout<<"get H5Pget_cache groupHandle_ "<<ret<<"\n";
            return CacheReturnType(somePrime, nBytes, rdcc); 
        });


    }

}
}


#endif
