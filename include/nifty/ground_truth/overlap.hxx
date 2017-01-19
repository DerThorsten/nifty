#ifndef NIFTY_TOOLS_OVERLAP_HXX
#define NIFTY_TOOLS_OVERLAP_HXX

#include <vector>
#include <unordered_map>

#include "nifty/marray/marray.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace ground_truth{

    template<class LABEL_TYPE = uint32_t, class COUNT_TYPE = uint32_t>
    class Overlap{
    public:

        typedef LABEL_TYPE LabelType;
        typedef COUNT_TYPE  CountType;
        typedef std::unordered_map<LabelType, CountType> MapType;

        template<class SET_A_ITER, class SET_B_ITER>
        Overlap(
            const uint64_t maxLabelSetA,
            SET_A_ITER aBegin,
            SET_A_ITER aEnd,
            SET_B_ITER bBegin
        )
        :   overlap_(maxLabelSetA+1),
            counts_(maxLabelSetA+1){

            fill(aBegin, aEnd, bBegin);
        }

        template<class LABEL_A, class LABEL_B>
        Overlap(
            const uint64_t maxLabelSetA,
            const marray::View<LABEL_A> arrayA,
            const marray::View<LABEL_B> arrayB
        )
        :   overlap_(maxLabelSetA+1),
            counts_(maxLabelSetA+1){

            const auto dimA = arrayA.dimension();
            const auto dimB = arrayB.dimension();
            NIFTY_CHECK_OP(dimA,==,dimB,"dimension mismatch in Overlap::Overlap")

            for(auto d=0; d<dimA; ++d){
                NIFTY_CHECK_OP(arrayA.shape(d),==,arrayB.shape(d),"shape mismatch in Overlap::Overlap")
            }

            if(dimA == 1){
                fill<1>(arrayA, arrayB);
            }
            if(dimA == 2){
                fill<2>(arrayA, arrayB);
            }
            if(dimA == 3){
                fill<2>(arrayA, arrayB);
            }
            if(dimA == 4){
                fill<2>(arrayA, arrayB);
            }
            else{
                auto aBegin = arrayA.begin();
                auto aEnd = arrayA.end();
                auto bBegin = arrayB.begin();
                fill(aBegin, aEnd, bBegin);
            }

        }


        double differentOverlap(const LabelType u, const LabelType v)const{

            const auto & olU = overlap_[u];
            const auto & olV = overlap_[v];
            const auto sU = float(counts_[u]);
            const auto sV = float(counts_[v]);
            auto isDiff = 0.0;
            for(const auto & keyAndSizeU : olU)
            for(const auto & keyAndSizeV : olV){

                auto keyU =  keyAndSizeU.first;
                auto rSizeU = float(keyAndSizeU.second)/sU;
                auto keyV =  keyAndSizeV.first;
                auto rSizeV = float(keyAndSizeV.second)/sV;

                if(keyU != keyV){
                    isDiff += (rSizeU * rSizeV);
                }
            }
            return isDiff;  
        }

        double bleeding(const LabelType u)const{
            const COUNT_TYPE size = counts_[u];
            const auto & ol = overlap_[u];

            std::vector<COUNT_TYPE> olCount;
            olCount.reserve(ol.size());

            COUNT_TYPE maxOlCount = 0;

            for(const auto & kv : ol){
                maxOlCount = std::max(maxOlCount, kv.second);
            }
            
            return 1.0 - (double(size) - double(maxOlCount))/size;

        }
        const std::vector<CountType> & counts()const{
            return counts_;
        };
        const std::vector<MapType> & overlap()const{
            return overlap_;
        };
    private:

        template<class SET_A_ITER, class SET_B_ITER>
        void fill(
            SET_A_ITER aBegin,
            SET_A_ITER aEnd,
            SET_B_ITER bBegin
        ){

            while(aBegin != aEnd){

                const auto labelA = *aBegin;
                const auto labelB = *bBegin;

                ++overlap_[labelA][labelB];
                ++counts_[labelA];
                ++aBegin;
                ++bBegin;
            }
        }


        template<size_t DIM, class LABEL_A, class LABEL_B>
        void fill(
            const marray::View<LABEL_A> arrayA,
            const marray::View<LABEL_B> arrayB
        ){
            typedef array::StaticArray<int64_t, DIM> Coord;

            Coord shape;
            for(auto d=0; d<DIM; ++d){
                    shape[d] = arrayA.shape(d);
            }
            tools::forEachCoordinate(shape,[&](const Coord coord){
                const auto la = arrayA(coord.asStdArray());
                ++overlap_[la][arrayB(coord.asStdArray())];
                ++counts_[la];
            });
        }
        std::vector<CountType> counts_;
        std::vector<MapType>   overlap_;
    };


} // end namespace nifty::ground_truth
} // end namespace nifty

#endif /*NIFTY_TOOLS_OVERLAP_HXX*/
