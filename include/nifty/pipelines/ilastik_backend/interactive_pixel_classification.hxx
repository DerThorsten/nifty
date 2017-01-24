#pragma once

#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/block_access.hxx"

namespace nifty{
namespace pipelines{
namespace ilastik_backend{


    
    template<class T, size_t DIM, bool MULTICHANNEL>
    class InputDataBase{
    public:
        typedef array::StaticArray<int64_t, DIM> Coord;
        virtual ~InputDataBase(){}

        virtual void readData(const Coord & begin, const Coord & end, nifty::marray::View<T> & out) = 0;
    };  



    template<class T, size_t DIM, bool MULTICHANNEL, class INTERNAL_TYPE>
    class Hdf5Input : public InputDataBase<T, DIM, MULTICHANNEL>
    {
    public:
        typedef InputDataBase<T, DIM, MULTICHANNEL> BaseType;
        typedef array::StaticArray<int64_t, DIM> Coord;

        virtual ~Hdf5Input(){};

        Hdf5Input(
            const nifty::hdf5::Hdf5Array<INTERNAL_TYPE> & data
        )
        :   BaseType(),
            data_(data){
        }

        virtual void readData(const Coord & begin, const Coord & end, nifty::marray::View<T> & out){
            nifty::marray::Marray<INTERNAL_TYPE> buffer(out.shapeBegin(), out.shapeEnd());
            mutex_.lock();
            data_.readSubarray(begin.begin(), buffer);
            mutex_.unlock();
            out = buffer;
        }
    private:
        std::mutex mutex_;
        const nifty::hdf5::Hdf5Array<INTERNAL_TYPE> & data_;
    };  

    


    template<
        class INPUT_TYPE_TAG,
        bool MULTICHANNEL
    >
    class InteractivePixelClassification{
    public:
        typedef INPUT_TYPE_TAG InputTypeTagType;
        typedef typename InputTypeTagType::DimensionType DimensionType;
        typedef InputDataBase<float, DimensionType::value, MULTICHANNEL> InputDataBaseType;

        size_t addTrainingInstance(const InputDataBaseType * instance){
            trainingData_.push_back(instance);
        }

        void addTrainingData(){
            
        }
    private:
        // atm we make it single channnel
        std::vector<const InputDataBaseType * > trainingData_;
    };  
}
}
}