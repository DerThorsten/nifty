#pragma once

#include <string>
#include <vector>

#include "nifty/tools/block_access.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/marray/marray_hdf5.hxx"

namespace nifty{
namespace hdf5{
    
    using namespace marray::hdf5;


    struct CacheSettings{

        CacheSettings( 
            const int   hashTabelSize_ = 977,
            const int   nBytes_ = 36000000,
            const float rddc_ = 1.0
        )
        : 
        hashTabelSize( hashTabelSize_),
        nBytes( nBytes_),
        rddc( rddc_){
        }

        int hashTabelSize;
        int nBytes;
        float rddc;
    };


    inline hid_t
    createFile
    (
        const std::string& filename,
        const CacheSettings & cacheSettings,
        HDF5Version hdf5version = DEFAULT_HDF5_VERSION
    )
    {
        auto plist = H5Pcreate(H5P_FILE_ACCESS);
        if(hdf5version == LATEST_HDF5_VERSION) {
            H5Pset_libver_bounds(plist, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
        }

        auto ret = H5Pset_cache(plist, 0.0, cacheSettings.hashTabelSize,  cacheSettings.nBytes, cacheSettings.rddc);

        hid_t fileHandle = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist);
        if(fileHandle < 0) {
            throw std::runtime_error("Could not create HDF5 file: " + filename);
        }

        return fileHandle;
    }

    /// Open an HDF5 file.
    ///
    /// \param filename Name of the file.
    /// \param fileAccessMode File access mode.
    /// \param hdf5version HDF5 version tag.
    ///
    /// \returns HDF5 handle
    ///
    /// \sa closeFile(), createFile()
    ///
    inline hid_t
    openFile
    (
        const std::string& filename,
        const CacheSettings & cacheSettings,
        FileAccessMode fileAccessMode = READ_ONLY,
        HDF5Version hdf5version = DEFAULT_HDF5_VERSION
    )
    {
        hid_t access = H5F_ACC_RDONLY;
        if(fileAccessMode == READ_WRITE) {
            access = H5F_ACC_RDWR;
        }


        auto plist = H5Pcreate(H5P_FILE_ACCESS);
        if(hdf5version == LATEST_HDF5_VERSION) {
            H5Pset_libver_bounds(plist, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
        }

        auto ret = H5Pset_cache(plist, 0.0, cacheSettings.hashTabelSize,  cacheSettings.nBytes, cacheSettings.rddc);

        hid_t fileHandle = H5Fopen(filename.c_str(), access, plist);
        if(fileHandle < 0) {
            throw std::runtime_error("Could not open HDF5 file: " + filename);
        }

        return fileHandle;
    }









} // namespace nifty::hdf5
} // namespace nifty

