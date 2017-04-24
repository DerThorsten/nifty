
namespace nifty{
namespace histogram{


    template<class T, class BINCOUNT=float>
    class Histogram{
    public:
        typedef BINCOUNT BincountType;
        Histogram(
            const T minVal, 
            const T maxVal,
            const size_t bincount
        )
        :   counts_(bincount),
            minVal_(minVal),
            maxVal_(maxVal),
            sum_(0)
        {
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
                wLow  = high - b;
                wHigh = double(b) - low;

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

        template<class U, size_t N>
        std::array<double, N>  quantiles(
            std::array<U, N> q
        )const{

            std::array<double, N> ret;
            std::array<double, N> qn;

            for(auto i=0; i<N; ++i){
                qn[i] = q[i] * this->sum_; 
            }

            double csum = 0.0;
            auto qi = 0;
            for(auto bin=0; bin<counts_.size(); ++bin){

                const double newcsum = csum  + counts_[i];
                while(qi < N && csum <= qn[qi] && newcsum >= qn[qi] ){
                    if(bin == 0 ){
                        ret[qi] = fbinToValue(0.0);
                    }
                    // linear interpolate the bin index    
                    else{
                        const auto lbin  = double(bin - 1);
                        const auto hbin =  double(bin);
                        const auto m = counts_[i];
                        const auto c = newcsum - hbin*m;
                        ret[qi] = fbinToValue((qn[qi] - c)/m);
                    }
                    ++qi;
                }
                csum = newcsum;
            }
        }

    private:

        double fbinToValue(const double fbin){

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
            val =/ (maxVal_ - minVal_);

            return val*(this->numberOfBins()-1);
        }


        std::vector<F_COUNT> counts_:
        T minVal_;
        T maxVal_;
        F_COUNT sum_;
    };





}
}