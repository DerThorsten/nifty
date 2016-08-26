#pragma once
#ifndef NIFTY_HDF5_HDF5_ARRAY
#define NIFTY_HDF5_HDF5_ARRAY

#include <string>
#include <vector>

#include "nifty/tools/block_access.hxx"
#include "nifty/tools/runtime_check.hxx"
#include "nifty/marray/marray_hdf5.hxx"

namespace nifty{
namespace hdf5{
    
    using namespace marray::hdf5;




    template<class T>
    class Hdf5Array{
    public:
        template<class SHAPE_ITER, class CHUNK_SHAPE_ITER>
        Hdf5Array(
            const hid_t& groupHandle,
            const std::string & datasetName,
            SHAPE_ITER shapeBegin,
            SHAPE_ITER shapeEnd,
            CHUNK_SHAPE_ITER chunkShapeBegin
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

            // dataset shape
            auto dataspace = H5Screate_simple(hsize_t(dim), shape.data(), NULL);

            // create the dataset
            dataset_ = H5Dcreate(groupHandle_, datasetName.c_str(), datatype_, dataspace, 
                        H5P_DEFAULT,dcplId, H5P_DEFAULT);

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
                throw std::runtime_error("Marray cannot open dataset.");
            }



            /*
            auto plist = H5Dget_access_plist(dataset_);

            //H5Pset_cache(hid_t plist_id, int mdc_nelmts, size_t rdcc_nslots, size_t rdcc_nbytes, double rdcc_w0)

            const auto anyVal = 0;
            const auto somePrime = 977;
            const auto nBytes = 36000000;
            const auto rddc = 1.0;
            auto ret = H5Pset_cache(plist, anyVal, somePrime,  nBytes, rddc);

            H5Pclose(plist);
            std::cout<<"set H5Pset_cache groupHandle_ "<<ret<<"\n";

            */


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
        uint64_t shape(const size_t d)const{
            return shape_[d];
        }
        const std::vector<uint64_t> & shape()const{
            return shape_;
        }
        uint64_t chunkShape(const size_t d)const{
            return chunkShape_[d];
        }
        const std::vector<uint64_t> & chunkShape()const{
            return chunkShape_;
        }

        bool isChunked()const{
            return isChunked_;
        }

        template<class ITER>
        void readSubarray(
            ITER roiBeginIter,
            marray::View<T> & out
        )const{
            NIFTY_CHECK_OP(out.dimension(),==,this->dimension(),"out has wrong dimension");
            NIFTY_CHECK(out.coordinateOrder() == marray::FirstMajorOrder, 
                "currently only views with last major order are supported"
            );
            //std::cout<<"load hyperslab\n";
            this->loadHyperslab(roiBeginIter, roiBeginIter+out.dimension(), out.shapeBegin(), out);
        }

        template<class ITER>
        void writeSubarray(
            ITER roiBeginIter,
            const marray::View<T> & in
        ){
            NIFTY_CHECK_OP(in.dimension(),==,this->dimension(),"in has wrong dimension");
            NIFTY_CHECK(in.coordinateOrder() == marray::FirstMajorOrder, 
                "currently only views with last major order are supported"
            );
            this->saveHyperslab(roiBeginIter, roiBeginIter+in.dimension(), in.shapeBegin(), in);
        }

    private:
        template<class BaseIterator, class ShapeIterator>
        void loadHyperslab(
            BaseIterator baseBegin,
            BaseIterator baseEnd,
            ShapeIterator shapeBegin,
            marray::View<T> & out
        ) const {
            HandleCheck<marray::MARRAY_NO_DEBUG> handleCheck;

            // determine shape of hyperslab and array
            std::size_t size = std::distance(baseBegin, baseEnd);
            std::vector<hsize_t> offset(size);
            std::vector<hsize_t> slabShape(size);
            std::vector<hsize_t> marrayShape(size);
            marray::CoordinateOrder coordinateOrder;
            if(H5Aexists(dataset_, reverseShapeAttributeName) > 0) {
                NIFTY_CHECK(false, "currently we do not allow to load from datasets with reverseShapeAttribute")
            } 
            else {
                // don't reverse base and shape
                coordinateOrder = marray::FirstMajorOrder;
                for(std::size_t j=0; j<size; ++j) {
                    offset[j] = hsize_t(*baseBegin);
                    slabShape[j] = hsize_t(*shapeBegin);
                    marrayShape[j] = slabShape[j];

                    ++baseBegin;
                    ++shapeBegin;
                }
            }
            
            hid_t dataspace = H5Dget_space(dataset_);
            herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, 
                &offset[0], NULL, &slabShape[0], NULL);
            if(status < 0) {
                H5Sclose(dataspace);
                throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape !");
            }

            // select memspace hyperslab
            hid_t memspace = H5Screate_simple(int(size), &marrayShape[0], NULL);
            std::vector<hsize_t> offsetOut(size, 0); // no offset
            status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &offsetOut[0],
                NULL, &marrayShape[0], NULL);
            if(status < 0) {
                H5Sclose(memspace); 
                H5Sclose(dataspace);
                throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape s!");
            }

            // read from dataspace into memspace
            //out = Marray<T>(SkipInitialization, &marrayShape[0], 
            //    (&marrayShape[0])+size, coordinateOrder);
            
            if(out.isSimple()){
                //std::cout<<"is simple\n";
                status = H5Dread(dataset_, datatype_, memspace, dataspace,
                    H5P_DEFAULT, &(out(0)));

                //std::cout<<"read status "<<status<<"\n";
            }
            else{
                //std::cout<<"is not simple\n";
                marray::Marray<T> tmpOut(marray::SkipInitialization, &marrayShape[0], 
                    (&marrayShape[0])+size, coordinateOrder);

                status = H5Dread(dataset_, datatype_, memspace, dataspace,
                    H5P_DEFAULT, &(tmpOut(0)));

                //std::cout<<"read status "<<status<<"\n";

                out = tmpOut;
            }

            // clean up
            H5Sclose(memspace); 
            H5Sclose(dataspace);
            if(status < 0) {
                throw std::runtime_error("Marray cannot read from dataset.");
            }
            handleCheck.check();
        }

        template<class BaseIterator, class ShapeIterator>
        void 
        saveHyperslab(
            BaseIterator baseBegin,
            BaseIterator baseEnd,
            ShapeIterator shapeBegin,
            const marray::View<T> & in
        ) {
            HandleCheck<marray::MARRAY_NO_DEBUG> handleCheck;

            // determine hyperslab shape
            std::vector<hsize_t> memoryShape(in.dimension());
            for(std::size_t j=0; j<in.dimension(); ++j) {
                memoryShape[j] = in.shape(j);
            }
            std::size_t size = std::distance(baseBegin, baseEnd);
            std::vector<hsize_t> offset(size);
            std::vector<hsize_t> slabShape(size);
            bool reverseShapeAttribute = (H5Aexists(dataset_, reverseShapeAttributeName) > 0);
            if(reverseShapeAttribute || in.coordinateOrder() == marray::LastMajorOrder) {
                NIFTY_CHECK(false, "neither reverseShapeAttribute nor LastMajorOrder are currently supported");
            }
            else{
               for(std::size_t j=0; j<size; ++j) {
                   offset[j] = hsize_t(*baseBegin);
                   slabShape[j] = hsize_t(*shapeBegin);
                   ++baseBegin;
                   ++shapeBegin;
               }
            }

            // select dataspace hyperslab
            hid_t dataspace = H5Dget_space(dataset_);
            herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, 
                &offset[0], NULL, &slabShape[0], NULL);
            if(status < 0) {
                H5Sclose(dataspace);
                H5Dclose(dataset_);
                throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape!");
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
                throw std::runtime_error("Marray cannot select hyperslab. Check offset and shape!");
            }

            if(in.isSimple()){
               status = H5Dwrite(dataset_, datatype_, memspace, dataspace, H5P_DEFAULT, &(in(0)));
            }
            else{
                marray::Marray<T> tmp(in);
                status = H5Dwrite(dataset_, datatype_, memspace, dataspace, H5P_DEFAULT, &(tmp(0)));
            }
            // clean up
            H5Sclose(memspace); 
            H5Sclose(dataspace);
            if(status < 0) {
                throw std::runtime_error("Marray cannot write to dataset.");
            }
            handleCheck.check();
        }

        void loadShape(std::vector<uint64_t> & shapeVec){

            marray::marray_detail::Assert(marray::MARRAY_NO_ARG_TEST || groupHandle_ >= 0);
            HandleCheck<marray::MARRAY_NO_DEBUG> handleCheck;

            hid_t filespace = H5Dget_space(dataset_);
            hsize_t dimension = H5Sget_simple_extent_ndims(filespace);
            hsize_t* shape = new hsize_t[(std::size_t)(dimension)];
            herr_t status = H5Sget_simple_extent_dims(filespace, shape, NULL);
            if(status < 0) {
                H5Sclose(filespace);
                delete[] shape;
                throw std::runtime_error("MarrayNifty cannot get extension of dataset.");
            }
            // write shape to shape_
            shapeVec.resize(dimension);
            if(H5Aexists(dataset_, reverseShapeAttributeName) > 0) {
                NIFTY_CHECK(false, "currently we do not allow to load from datasets with reverseShapeAttribute")
            }
            else {
                for(std::size_t j=0; j<shapeVec.size(); ++j) {
                    shapeVec[j] = uint64_t(shape[j]);
                }
            }
            // clean up
            delete[] shape;
            H5Sclose(filespace);
            handleCheck.check();
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


    namespace tools{

}


} // namespace nifty::hdf5



namespace tools{

    template<class T, class COORD>
    inline void readSubarray(
        const hdf5::Hdf5Array<T> array,
        const COORD & beginCoord,
        const COORD & endCoord,
        marray::View<T> & subarray
    ){
        array.readSubarray(beginCoord.begin(), subarray);
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

#endif  // NIFTY_HDF5_HDF5_ARRAY
