#ifndef NIFTY_TOOLS_OVERLAP_HXX
#define NIFTY_TOOLS_OVERLAP_HXX

#include <vector>
#include <unordered_map>

namespace nifty{
namespace tools{

    template<class LABEL_TYPE = uint32_t, class COUNT_TYPE = uint32_t>
    class Overlap{
    public:

        typedef LABEL_TYPE LabelType;
        typedef CountType  CountType;
        typedef std::unordered_map<LabelType, CountType> MapType;

        template<class SET_A_ITER, class SET_B_ITER>
        Overlap(
            const uint64_t maxLabelSetA,
            SET_A_ITER aBegin,
            SET_A_ITER aEnd,
            SET_B_ITER bBegin
        )
        :   overlap_(maxLabelSetA){

            while(aBegin != aEnd){

                const auto labelA = *aBegin;
                const auto labelB = *bBegin;

                ++overlap_[labelA][labelB];

                ++aBegin;
                ++bBegin;
            }
        }


        double differentOverlap(const LabelType u, const LabelType v)const{

            const auto & olU = overlap_[u];
            const auto & olV = overlap_[v];
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

    private:


        std::vector<MapType> overlap_;
    };


} // end namespace nifty::tools
} // end namespace nifty

#endif /*NIFTY_TOOLS_OVERLAP_HXX*/
