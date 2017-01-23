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
    };  



    template<class T, size_t DIM, bool MULTICHANNEL>
    class Hdf5InputBase : InputDataBase<T, DIM, MULTICHANNEL>
    {
    public:
        typedef InputDataBase<T, DIM, MULTICHANNEL> BaseType;
        typedef array::StaticArray<int64_t, DIM> Coord;

        virtual ~Hdf5InputBase(){};

        Hdf5InputBase(
            const nifty::hdf5::Hdf5Array<T> & data
        )
        :   BaseType(),
            data_(data)
        {

        }
    private:
        const nifty::hdf5::Hdf5Array<T> & data_;
    };  

    


    template<
        class INPUT_TYPE_TAG
    >
    class InteractivePixelClassification{
    public:
        
    private:

    };  
}
}
}