#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/array/arithmetic_array.hxx"

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
        typedef array::StaticArray<int64_t, DIM> Coord;
        virtual ~Hdf5InputBase(){};
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