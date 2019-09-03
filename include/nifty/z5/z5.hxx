#pragma once

#include "z5/multiarray/xtensor_access.hxx"
#include "z5/factory.hxx"

namespace nifty{
namespace nz5 {

    template<class T>
    class DatasetWrapper {

    public:
        typedef z5::filesystem::Dataset<T> WrappedType;
        typedef z5::Dataset WrappedBaseType;
        typedef std::shared_ptr<WrappedBaseType> PointerType;
        typedef typename WrappedType::value_type value_type;
        typedef typename WrappedType::shape_type shape_type;

        DatasetWrapper(const std::string & pathToFile, const std::string & key) {
            z5::filesystem::handle::File file(pathToFile);
            wrappedPtr_ = std::move(z5::openDataset(file, key));
        }

        const WrappedBaseType & wrapped() const {return *wrappedPtr_;}
        WrappedBaseType & wrapped() {return *wrappedPtr_;}
        const shape_type & shape() const {return wrappedPtr_->shape();}
    private:
        PointerType wrappedPtr_;
    };

}

namespace tools{

    template<class ARRAY, class COORD>
    inline void readSubarray(const nz5::DatasetWrapper<typename ARRAY::value_type> & ds,
                             const COORD & beginCoord,
                             const COORD & endCoord,
                             xt::xexpression<ARRAY> & subarray){
        z5::multiarray::readSubarray<typename ARRAY::value_type>(ds.wrapped(), subarray, beginCoord.begin());
    }

    template<class ARRAY, class COORD>
    inline void writeSubarray(nz5::DatasetWrapper<typename ARRAY::value_type> & ds,
                              const COORD & beginCoord,
                              const COORD & endCoord,
                              const xt::xexpression<ARRAY> & subarray){
        z5::multiarray::writeSubarray<typename ARRAY::value_type>(ds.wrapped(), subarray, beginCoord.begin());
    }

    template<class T>
    inline bool isChunked(const nz5::DatasetWrapper<T> & ds) {
        return true;
    }

    template<class T>
    inline std::vector<std::size_t> getChunkShape(const nz5::DatasetWrapper<T> & ds) {
        return ds.wrapped().defaultChunkShape();
    }

} // namespace nifty::tools
} // namespace nifty
