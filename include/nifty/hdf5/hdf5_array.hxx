#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "xtensor/xarray.hpp"

#include "nifty/hdf5/hdf5.hxx"
#include "nifty/tools/block_access.hxx"
#include "nifty/tools/runtime_check.hxx"

namespace nifty{
namespace hdf5{

    template<class T>
    class Hdf5Array{
    public:

        typedef T DataType;
        typedef T value_type;

        template<class SHAPE_ITER, class CHUNK_SHAPE_ITER>
        Hdf5Array(
            const hid_t& groupHandle,
            const std::string & datasetName,
            SHAPE_ITER shapeBegin,
            SHAPE_ITER shapeEnd,
            CHUNK_SHAPE_ITER chunkShapeBegin,
            const int compression = -1 // -1 means no compression
        )
        :   groupHandle_(groupHandle),
            dataset_(),
            datatype_(),
            isChunked_(true)
        {
            datatype_ = H5Tcopy(hdf5Type<T>());
            const auto dim = std::distance(shapeBegin, shapeEnd);

            shape_.resize(dim);
            chunkShape_.resize(dim);

            std::vector<hsize_t> shape(dim);
            std::vector<hsize_t> chunkShape(dim);

            for(auto d=0; d<dim; ++d){
                const auto s = *shapeBegin;
                const auto cs = *chunkShapeBegin;
                shape[d] = s;
                shape_[d] = s;
                chunkShape[d] = cs;
                chunkShape_[d] = cs;
                ++shapeBegin;
                ++chunkShapeBegin;
            }

            // chunk properties
            hid_t dcplId = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcplId, hsize_t(dim), chunkShape.data());

            if(compression > 0){
                if(!H5Zfilter_avail(H5Z_FILTER_DEFLATE)){
                    throw std::runtime_error("gzip filter not available");
                }
                else{
                    const auto status = H5Pset_deflate(dcplId, compression);
                    if(status!=0){
                        throw std::runtime_error("error in  H5Pset_deflate");
                    }
                }
            }


            // dataset shape
            auto dataspace = H5Screate_simple(hsize_t(dim), shape.data(), NULL);

            // create the dataset
            dataset_ = H5Dcreate(groupHandle_, datasetName.c_str(), datatype_, dataspace,
                                 H5P_DEFAULT, dcplId, H5P_DEFAULT);

            // close the dataspace and the chunk properties
            H5Sclose(dataspace);
            H5Pclose(dcplId);
        }


        // constructor for new array without chunks and compression
        template<class SHAPE_ITER>
        Hdf5Array(
            const hid_t& groupHandle,
            const std::string & datasetName,
            SHAPE_ITER shapeBegin,
            SHAPE_ITER shapeEnd
        )
        :   groupHandle_(groupHandle),
            dataset_(),
            datatype_(),
            isChunked_(false)
        {
            datatype_ = H5Tcopy(hdf5Type<T>());
            const auto dim = std::distance(shapeBegin, shapeEnd);

            shape_.resize(dim);
            chunkShape_.resize(dim);

            std::vector<hsize_t> shape(dim);
            std::vector<hsize_t> chunkShape(dim);

            for(auto d=0; d<dim; ++d){
                const auto s = *shapeBegin;
                shape[d] = s;
                shape_[d] = s;
                chunkShape_[d] = s;
                ++shapeBegin;
            }

            // chunk properties
            hid_t dcplId = H5Pcreate(H5P_DATASET_CREATE);

            // dataset shape
            auto dataspace = H5Screate_simple(hsize_t(dim), shape.data(), NULL);

            // create the dataset
            dataset_ = H5Dcreate(groupHandle_, datasetName.c_str(), datatype_, dataspace,
                                 H5P_DEFAULT, dcplId, H5P_DEFAULT);

            // close the dataspace and the chunk properties
            H5Sclose(dataspace);
            H5Pclose(dcplId);
        }

        Hdf5Array(
            const hid_t& groupHandle,
            const std::string & datasetName
        )
        :   groupHandle_(groupHandle),
            dataset_(),
            datatype_(),
            isChunked_(true)
        {

            dataset_ = H5Dopen(groupHandle_, datasetName.c_str(), H5P_DEFAULT);
            if(dataset_ < 0) {
                throw std::runtime_error("Ccannot open dataset.");
            }

            // select dataspace hyperslab
            datatype_ = H5Dget_type(dataset_);
            if(!H5Tequal(datatype_, hdf5Type<T>())) {
                throw std::runtime_error("data type of stored hdf5 dataset and passed array do not match in loadHyperslab");
            }


            this->loadShape(shape_);
            this->loadChunkShape(chunkShape_);
        }

        int setCache(){
            //herr_t H5Pset_cache(hid_t plist_id, int mdc_nelmts, size_t rdcc_nslots, size_t rdcc_nbytes, double rdcc_w0)
        }

        ~Hdf5Array(){
            H5Tclose(datatype_);
            H5Dclose(dataset_);
        }

        uint64_t dimension()const{
            return shape_.size();
        }

        uint64_t shape(const std::size_t d)const{
            return shape_[d];
        }

        const std::vector<uint64_t> & shape()const{
            return shape_;
        }

        uint64_t chunkShape(const std::size_t d)const{
            return chunkShape_[d];
        }

        const std::vector<uint64_t> & chunkShape()const{
            return chunkShape_;
        }

        bool isChunked()const{
            return isChunked_;
        }

        template<class ITER, class ARRAY>
        void readSubarray(
            ITER roiBeginIter,
            ARRAY & out
        )const{
            NIFTY_CHECK_OP(out.dimension(),==,
                           this->dimension(),
                           "out has wrong dimension");
            this->loadHyperslab(roiBeginIter,
                                roiBeginIter + out.dimension(),
                                out.shape().begin(), out);
        }

        template<class ITER, class ARRAY>
        void writeSubarray(
            ITER roiBeginIter,
            const ARRAY & in
        ){
            NIFTY_CHECK_OP(in.dimension(),==,this->dimension(),"in has wrong dimension");
            this->saveHyperslab(roiBeginIter,
								roiBeginIter + in.dimension(),
								in.shape().begin(), in);
        }

    private:
        template<class BaseIterator, class ShapeIterator, class ARRAY>
        void loadHyperslab(
            BaseIterator baseBegin,
            BaseIterator baseEnd,
            ShapeIterator shapeBegin,
            ARRAY & out
        ) const {
            // determine shape of hyperslab and array
            std::size_t size = std::distance(baseBegin, baseEnd);
            std::vector<hsize_t> offset(size);
            std::vector<hsize_t> slabShape(size);
            std::vector<hsize_t> arrayShape(size);

            for(std::size_t j = 0; j < size; ++j) {
                offset[j] = hsize_t(*baseBegin);
                slabShape[j] = hsize_t(*shapeBegin);
                arrayShape[j] = slabShape[j];

                ++baseBegin;
                ++shapeBegin;
            }

            hid_t dataspace = H5Dget_space(dataset_);
            if(dataspace < 0) {
                throw std::runtime_error("Can't open dataspace!");
            }

            herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,
                &offset[0], NULL, &slabShape[0], NULL);
            if(status < 0) {
                H5Sclose(dataspace);
                throw std::runtime_error("Cannot select hyperslab. Check offset and shape !");
            }

            // select memspace hyperslab
            hid_t memspace = H5Screate_simple(int(size), &arrayShape[0], NULL);
            std::vector<hsize_t> offsetOut(size, 0); // no offset
            status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &offsetOut[0],
                NULL, &arrayShape[0], NULL);
            if(status < 0) {
                H5Sclose(memspace);
                H5Sclose(dataspace);
                throw std::runtime_error("Cannot select hyperslab. Check offset and shape s!");
            }

            // read from dataspace into memspace
            status = H5Dread(dataset_, datatype_, memspace, dataspace,
                             H5P_DEFAULT, &(out(0)));

            // clean up
            H5Sclose(memspace);
            H5Sclose(dataspace);
            if(status < 0) {
                throw std::runtime_error("Cannot read from dataset.");
            }
        }

        template<class BaseIterator, class ShapeIterator, class ARRAY>
        void
        saveHyperslab(
            BaseIterator baseBegin,
            BaseIterator baseEnd,
            ShapeIterator shapeBegin,
            const ARRAY & in
        ) {
            // determine hyperslab shape
            std::vector<hsize_t> memoryShape(in.dimension());
            for(std::size_t j=0; j<in.dimension(); ++j) {
                memoryShape[j] = in.shape()[j];
            }
            std::size_t size = std::distance(baseBegin, baseEnd);
            std::vector<hsize_t> offset(size);
            std::vector<hsize_t> slabShape(size);

            for(std::size_t j=0; j<size; ++j) {
                offset[j] = hsize_t(*baseBegin);
                slabShape[j] = hsize_t(*shapeBegin);
                ++baseBegin;
                ++shapeBegin;
            }

            // select dataspace hyperslab
            hid_t dataspace = H5Dget_space(dataset_);
            herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET,
                &offset[0], NULL, &slabShape[0], NULL);
            if(status < 0) {
                H5Sclose(dataspace);
                H5Dclose(dataset_);
                throw std::runtime_error("Cannot select hyperslab. Check offset and shape!");
            }

            // select memspace hyperslab
            hid_t memspace = H5Screate_simple(int(in.dimension()), &memoryShape[0], NULL);
            std::vector<hsize_t> memoryOffset(int(in.dimension()), 0); // no offset
            status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &memoryOffset[0], NULL,
                &memoryShape[0], NULL);
            if(status < 0) {
                H5Sclose(memspace); ;
                H5Sclose(dataspace);
                H5Dclose(dataset_);
                throw std::runtime_error("Cannot select hyperslab. Check offset and shape!");
            }

            status = H5Dwrite(dataset_, datatype_,
                              memspace, dataspace, H5P_DEFAULT, &(in(0)));
            // clean up
            H5Sclose(memspace);
            H5Sclose(dataspace);
            if(status < 0) {
                throw std::runtime_error("Cannot write to dataset.");
            }
        }

        void loadShape(std::vector<uint64_t> & shapeVec){

            hid_t filespace = H5Dget_space(dataset_);
            hsize_t dimension = H5Sget_simple_extent_ndims(filespace);
            hsize_t* shape = new hsize_t[(std::size_t)(dimension)];
            herr_t status = H5Sget_simple_extent_dims(filespace, shape, NULL);
            if(status < 0) {
                H5Sclose(filespace);
                delete[] shape;
                throw std::runtime_error("nifty cannot get extension of dataset.");
            }
            // write shape to shape_
            shapeVec.resize(dimension);
            for(std::size_t j=0; j<shapeVec.size(); ++j) {
                shapeVec[j] = uint64_t(shape[j]);
            }
            // clean up
            delete[] shape;
            H5Sclose(filespace);
        }

        void loadChunkShape(std::vector<uint64_t> & chunkShape){
            const auto d = this->dimension();
            std::vector<hsize_t> chunkShapeTmp(d);
            auto plist = H5Dget_create_plist(dataset_);


            bool couldBeChunked = false;
            const auto layout =  H5Pget_layout(plist);
            if(layout == H5D_CHUNKED){
                couldBeChunked = true;
            }
            if(couldBeChunked){
                herr_t status = H5Pget_chunk(plist, int(d), chunkShapeTmp.data());
                if(status < 0) {
                    isChunked_ = false;
                    chunkShape_ = shape_;
                    //H5Pclose(plist);
                    //throw std::runtime_error("Nifty cannot get chunkShape of dataset");
                }
                else{
                    isChunked_ = true;
                    chunkShape_.resize(d);
                    std::copy(chunkShapeTmp.begin(), chunkShapeTmp.end(), chunkShape_.begin());
                }
            }
            else{
                isChunked_ = false;
                chunkShape_ = shape_;
            }
            H5Pclose(plist);
        }

        hid_t groupHandle_;
        hid_t dataset_;
        hid_t datatype_;
        std::vector<uint64_t> shape_;
        std::vector<uint64_t> chunkShape_;
        bool isChunked_;
    };
} // namespace nifty::hdf5



namespace tools{

    template<class T, class COORD, class ARRAY>
    inline void readSubarray(
        const hdf5::Hdf5Array<T> & array,
        const COORD & beginCoord,
        const COORD & endCoord,
        ARRAY & subarray
    ){
        array.readSubarray(beginCoord.begin(), subarray);
    }

    template<class T, class COORD, class ARRAY>
    inline void writeSubarray(
        hdf5::Hdf5Array<T> & array,
        const COORD & beginCoord,
        const COORD & endCoord,
        const ARRAY & subarray
    ){
        array.writeSubarray(beginCoord.begin(), subarray);
    }

    template<class ARRAY>
    struct BlockStorageSelector;

    template<class T>
    struct BlockStorageSelector<hdf5::Hdf5Array<T> >
    {
       typedef BlockStorage<T> type;
    };
} // namespace nifty::tools
} // namespace nifty
