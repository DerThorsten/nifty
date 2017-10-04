#pragma once

#include <vector>
#include <array>

namespace nifty{
namespace histogram{


    template<class T, class BINCOUNT=float>
    class Histogram{
    public:
        typedef BINCOUNT BincountType;
        Histogram(
            const T minVal=0, 
            const T maxVal=1,
            const size_t bincount=40
        )
        :   counts_(bincount,0),
            minVal_(minVal),
            maxVal_(maxVal),
            binWidth_((maxVal-minVal)/T(bincount)),
            sum_(0)
        {
        }
        template<class ITER>
        void clearSetMinMaxAndFillFrom(
            ITER begin,
            ITER end
        ){
            float minVal = std::numeric_limits<T>::infinity();
            float maxVal = static_cast<T>(-1.0) *  std::numeric_limits<T>::infinity();
            const auto size = std::distance(begin, end);
            for(auto i=0; i<size; ++i){
                minVal = std::min(minVal, begin[i]);
                maxVal = std::max(maxVal, begin[i]);
            }
            this->clear();
            this->setMinMax(minVal, maxVal);
            for(auto i=0; i<size; ++i){
               this->insert(begin[i]); 
            }
        }
        void setMinMax(
            const T minVal, 
            const T maxVal
        ){
            minVal_ = minVal;
            maxVal_ = maxVal;
        }


        const BincountType & operator[](const size_t i)const{
            return counts_[i];
        }
        size_t numberOfBins()const{
            return counts_.size();
        }
        BincountType sum()const{
            return sum_;
        }



        // insert     
        void insert(const T & value, const double w = 1.0){
            const auto b = this->fbin(value);
            const auto low  = std::floor(b);
            const auto high = std::ceil(b);

            // low and high are the same
            if(low + 0.5 >= high){
                counts_[size_t(low)] += w;
            }
            // low and high are different
            else{
                const auto wLow  = high - b;
                const auto wHigh = double(b) - low;

                counts_[size_t(low)]  += w*wLow;
                counts_[size_t(high)] += w*wHigh;
            }
            sum_ += w;
        }

        void normalize(){
            for(auto & v: counts_)
                v/=sum_;
            sum_ = 1.0;
        }

        void clear(){
            for(auto & v: counts_)
                v = 0;
            sum_ = 0.0;
        }

        double binToValue(const double fbin)const{
            return this->fbinToValue(fbin);
        }


        float binWidth()const{
            return binWidth_;
        }

        /// \warning this function has undefined behavior
        /// if bins and limits do not match
        void merge(const Histogram & other){
            for(auto b=0; b<counts_.size(); ++b){
                counts_[b] += other.counts_[b];
            }
            sum_ += other.sum_;
        }

    private:

        double fbinToValue(double fbin)const{
            //fbin += binWidth_/2.0;
            fbin /= double(counts_.size()-1);
            return (1.0-fbin)*minVal_  + fbin*maxVal_; 
        }

        /**
         * @brief      get the floating point bin index
         *
         * @param[in]  val   value which to put in a bin
         *
         * @return     the floating point bin in [0,numberOfBins()-1]
         */
        float fbin(T val)const{
            // truncate
            val = std::max(minVal_, val);
            val = std::min(maxVal_, val);

            // normalize
            val -= minVal_;
            val /= (maxVal_ - minVal_);

            return val*float(this->numberOfBins()-1);
        }



        std::vector<BincountType> counts_;
        T minVal_;
        T maxVal_;
        T binWidth_;
        BincountType sum_;
    };


    template<class HISTOGRAM, class RANK_ITER, class OUT_ITER>
    void quantiles(
        const HISTOGRAM & histogram,
        RANK_ITER ranksBegin,
        RANK_ITER ranksEnd,
        OUT_ITER outIter
    ){

        const auto nQuantiles = std::distance(ranksBegin, ranksEnd);
        const auto s = histogram.sum();

        //std::cout<<"nQuantiles "<<nQuantiles<<"\n";
        double csum = 0.0;
        auto qi = 0;
        for(auto bin=0; bin<histogram.numberOfBins(); ++bin){
            const double newcsum = csum  + histogram[bin];
            const auto  quant = ranksBegin[qi] * s;
            //std::cout<<"BIN "<<bin<<"\n";
            //std::cout<<"    qi "<<qi<<" quant "<<quant<<"\n";
            //std::cout<<"    csum "<<csum<<" newcsum "<<newcsum<<"\n";
            while(qi < nQuantiles && csum <= quant && newcsum >= quant ){
                if(bin == 0 ){
                    outIter[qi] = histogram.binToValue(0.0);
                }
                // linear interpolate the bin index    
                else{
                    //std::cout<<"        foundBIN\n";
                    const auto lbin  = double(bin-1) + histogram.binWidth()/2.0;
                    const auto hbin =  double(bin) + histogram.binWidth()/2.0;
                    const auto m = histogram[bin];
                    const auto c = csum - lbin*m;

                    //std::cout<<"        computed bin "<<(quant - c)/m<<"\n";

                    outIter[qi] = histogram.binToValue((quant - c)/m);
                }
                ++qi;
            }
            if(qi>=nQuantiles)
                break;
            csum = newcsum;
        }
    }


    




}
}