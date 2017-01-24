#pragma once

#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/blocking.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include "tbb/concurrent_lru_cache.h"
#include "tbb/concurrent_hash_map.h"

namespace nifty{
namespace pipelines{
namespace ilastik_backend{












    
    template<class T, size_t DIM, bool MULTICHANNEL>
    class InputDataBase{
    public:
        typedef array::StaticArray<int64_t, DIM> Coord;
        virtual ~InputDataBase(){}

        virtual void readData(const Coord & begin, const Coord & end, nifty::marray::View<T> & out) = 0;

        virtual Coord spaceTimeShape() const = 0;
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
        virtual Coord spaceTimeShape() const {
            const auto shape = data_.shape();
            NIFTY_CHECK_OP(shape.size(), ==, DIM," ");
            Coord c;
            std::copy(shape.begin(), shape.end(), c.begin());
            return c;
        }
    private:
        std::mutex mutex_;
        const nifty::hdf5::Hdf5Array<INTERNAL_TYPE> & data_;
    };  

        
    template<class INTERNAL_TYPE, size_t DIM, bool MULTICHANNEL>
    class Hdf5Input<INTERNAL_TYPE,DIM, MULTICHANNEL,INTERNAL_TYPE> : public InputDataBase<INTERNAL_TYPE, DIM, MULTICHANNEL>
    {
    public:
        typedef InputDataBase<INTERNAL_TYPE, DIM, MULTICHANNEL> BaseType;
        typedef array::StaticArray<int64_t, DIM> Coord;

        virtual ~Hdf5Input(){};

        Hdf5Input(
            const nifty::hdf5::Hdf5Array<INTERNAL_TYPE> & data
        )
        :   BaseType(),
            data_(data){
        }
        virtual void readData(const Coord & begin, const Coord & end, nifty::marray::View<INTERNAL_TYPE> & out){
            mutex_.lock();
            data_.readSubarray(begin.begin(), out);
            mutex_.unlock();
            }
        virtual Coord spaceTimeShape() const {
            const auto shape = data_.shape();
            NIFTY_CHECK_OP(shape.size(), ==, DIM," ");
            Coord c;
            std::copy(shape.begin(), shape.end(), c.begin());
            return c;
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
        typedef InputDataBase<uint8_t, DimensionType::value, MULTICHANNEL> InputDataBaseType;
        typedef array::StaticArray<int64_t, DimensionType::value> CoordType;


        struct BlockXY{
            std::vector<float> features_;
            std::vector<uint8_t> labels_;
        };
    



        InteractivePixelClassification(
            const InputDataBaseType *  trainingInstance,
            const size_t numberOfLabels,
            const array::StaticArray<int64_t, DimensionType::value> & blockSize
        )
        :   
            trainingInstance_(trainingInstance),
            numberOfLabels_(numberOfLabels),
            blockSize_(blockSize),    
            blocking_(CoordType(0), trainingInstance->spaceTimeShape(),blockSize)
        {
        }


        void addTrainingData(
            const nifty::marray::View<uint8_t> & labels,
            array::StaticArray<int64_t, DimensionType::value> coordBegin,
            array::StaticArray<int64_t, DimensionType::value> coordEnd
        ){  
            auto blockBegin = coordBegin / blocking_.blockShape();
            auto blockEnd   = coordEnd   / blocking_.blockShape() + CoordType(1);


            

            nifty::tools::forEachCoordinate(blockBegin,blockEnd,
                [&](const CoordType & blockCoord){
                    // shitty code.. to lazy to think about it
                    bool use = true;
                    for(auto d=0; d<DimensionType::value; ++d){
                        if(blockCoord[d] >= blocking_.blockShape()[d]){
                            use = false;
                            break;
                        }   
                    }
                    if(use){
                        const auto blockIndex = this->blockCoordToBlockIndex(blockCoord);
                        //  we need the features for this very block
                        // .... spawn in parallel
                        // X,Y
                        
                        // here we somehow get the feature Array for the complete block
                        // and extract X and Y at the very places where 
                        // we need it
                        
                    }
                }
            );

            // here we retrain 
            this->retrain();
            this->redoLastPrediction();
        }

        void retrain(){
            // make X and Y
        }

        void redoLastPrediction(){

        }

    private:

        size_t blockCoordToBlockIndex(const CoordType & blockCoord)const{
            auto i=0;
            for(auto d=0; d<DimensionType::value; ++d){
                i += blockCoord[d]*blocking_.blocksPerAxisStrides()[d];
            }
            return i;
        }

        const InputDataBaseType * trainingInstance_;
        uint8_t numberOfLabels_;
        array::StaticArray<int64_t, DimensionType::value> blockSize_;
        nifty::tools::Blocking<DimensionType::value, int64_t> blocking_; 



        tbb::concurrent_hash_map<size_t, BlockXY > blocksWithLabels_;

    };  
}
}
}