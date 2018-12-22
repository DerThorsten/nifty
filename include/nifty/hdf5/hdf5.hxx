// Based on code by Bjoern Andres with LICENSE
// Copyright (c) 2011-2013 by Bjoern Andres.
//
// This software was developed by Bjoern Andres.
// Enquiries shall be directed to bjoern@andres.sc.
//
// All advertising materials mentioning features or use of this software must
// display the following acknowledgement: ``This product includes
// andres::marray developed by Bjoern Andres. Please direct enquiries
// concerning andres::marray to bjoern@andres.sc''.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - All advertising materials mentioning features or use of this software must
//   display the following acknowledgement: ``This product includes
//   andres::marray developed by Bjoern Andres. Please direct enquiries
//   concerning andres::marray to bjoern@andres.sc''.
// - The name of the author must not be used to endorse or promote products
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <string>
#include <vector>
#include "hdf5.h"

#include "nifty/tools/block_access.hxx"
#include "nifty/tools/runtime_check.hxx"

namespace nifty{
namespace hdf5{

    enum FileAccessMode {READ_ONLY, READ_WRITE};
    enum HDF5Version {DEFAULT_HDF5_VERSION, LATEST_HDF5_VERSION};

    template<class T>
    inline hid_t uintTypeHelper() {
       switch(sizeof(T)) {
           case 1:
               return H5T_STD_U8LE;
           case 2:
               return H5T_STD_U16LE;
           case 4:
               return H5T_STD_U32LE;
           case 8:
               return H5T_STD_U64LE;
           default:
               throw std::runtime_error("No matching HDF5 type.");
       }
    }

    template<class T>
    inline hid_t intTypeHelper() {
       switch(sizeof(T)) {
           case 1:
               return H5T_STD_I8LE;
           case 2:
               return H5T_STD_I16LE;
           case 4:
               return H5T_STD_I32LE;
           case 8:
               return H5T_STD_I64LE;
           default:
               throw std::runtime_error("No matching HDF5 type.");
       }
    }

    template<class T>
    inline hid_t floatingTypeHelper() {
       switch(sizeof(T)) {
           case 4:
               return H5T_IEEE_F32LE;
           case 8:
               return H5T_IEEE_F64LE;
           default:
               throw std::runtime_error("No matching HDF5 type.");
       }
    }

    template<class T>
    inline hid_t hdf5Type();

    template<> inline hid_t hdf5Type<unsigned char>()
        { return uintTypeHelper<unsigned char>(); }
    template<> inline hid_t hdf5Type<unsigned short>()
        { return uintTypeHelper<unsigned short>(); }
    template<> inline hid_t hdf5Type<unsigned int>()
        { return uintTypeHelper<unsigned int>(); }
    template<> inline hid_t hdf5Type<unsigned long>()
        { return uintTypeHelper<unsigned long>(); }
    template<> inline hid_t hdf5Type<unsigned long long>()
        { return uintTypeHelper<unsigned long long>(); }

    template<> inline hid_t hdf5Type<signed char>()
        { return intTypeHelper<signed char>(); }
    template<> inline hid_t hdf5Type<char>()
        { return uintTypeHelper<char>(); }
    template<> inline hid_t hdf5Type<short>()
        { return intTypeHelper<short>(); }
    template<> inline hid_t hdf5Type<int>()
        { return intTypeHelper<int>(); }
    template<> inline hid_t hdf5Type<long>()
        { return intTypeHelper<long>(); }
    template<> inline hid_t hdf5Type<long long>()
        { return intTypeHelper<long long>(); }

    template<> inline hid_t hdf5Type<float>()
        { return floatingTypeHelper<float>(); }
    template<> inline hid_t hdf5Type<double>()
        { return floatingTypeHelper<double>(); }


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
        HDF5Version hdf5version = DEFAULT_HDF5_VERSION,
        const CacheSettings & cacheSettings=CacheSettings()
    )
    {
        auto plist = H5Pcreate(H5P_FILE_ACCESS);
        if(hdf5version == LATEST_HDF5_VERSION) {
            H5Pset_libver_bounds(plist, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
        }

        auto ret = H5Pset_cache(plist, 0.0, cacheSettings.hashTabelSize,
                                cacheSettings.nBytes, cacheSettings.rddc);

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
        FileAccessMode fileAccessMode=READ_ONLY,
        HDF5Version hdf5version=DEFAULT_HDF5_VERSION,
        const CacheSettings & cacheSettings=CacheSettings()
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

        auto ret = H5Pset_cache(plist, 0.0, cacheSettings.hashTabelSize,
                                cacheSettings.nBytes, cacheSettings.rddc);

        hid_t fileHandle = H5Fopen(filename.c_str(), access, plist);
        if(fileHandle < 0) {
            throw std::runtime_error("Could not open HDF5 file: " + filename);
        }

        return fileHandle;
    }


	inline void closeFile(const hid_t& handle){
	    H5Fclose(handle);
	}


	inline hid_t createGroup(const hid_t& parentHandle,
	    					 const std::string& groupName){
    	hid_t groupHandle = H5Gcreate(parentHandle, groupName.c_str(),
    	    						  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    	if(groupHandle < 0) {
    	    throw std::runtime_error("Could not create HDF5 group.");
    	}
    	return groupHandle;
	}


	inline hid_t openGroup(const hid_t& parentHandle,
	    				   const std::string& groupName){
	    hid_t groupHandle = H5Gopen(parentHandle, groupName.c_str(), H5P_DEFAULT);
	    if(groupHandle < 0) {
	        throw std::runtime_error("Could not open HDF5 group.");
	    }
	    return groupHandle;
	}


	inline void closeGroup(const hid_t& handle){
    	H5Gclose(handle);
	}
} // namespace nifty::hdf5
} // namespace nifty

