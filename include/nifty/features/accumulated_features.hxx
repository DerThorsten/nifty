#ifndef NIFTY_FEATURES_ACCUMULATED_FEATURES_HXX
#define NIFTY_FEATURES_ACCUMULATED_FEATURES_HXX



#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics.hpp>


namespace nifty{
namespace features{

    namespace bacc = boost::accumulators;

    template<class T>
    class DefaultAccumulatedStatistics{
    public:


        typedef bacc::accumulator_set<
            T, 
            bacc::stats<
                bacc::tag::count,
                bacc::tag::mean,
                bacc::tag::min, 
                bacc::tag::max,
                bacc::tag::moment<2>,
                bacc::tag::moment<3>,
                bacc::tag::tail_quantile<bacc::right>
            > 
        > AccType;

        typedef std::integral_constant<int, 1>  NPasses;
        typedef std::integral_constant<int, 11> NFeatures;

        DefaultAccumulatedStatistics(const size_t rightTailCacheSize = 1000)
        :   acc_(bacc::right_tail_cache_size = rightTailCacheSize){

        }
        DefaultAccumulatedStatistics & acc(const T & val, const size_t pass=0){
            acc_(val);
            return *this;
        }

        template<class RESULT_ITER>
        void result(RESULT_ITER rBegin, RESULT_ITER rEnd){
            using namespace boost::accumulators;
            const auto count = extract_result< tag::count>(acc_);
            const auto d = std::distance(rBegin,rEnd);
            NIFTY_ASSERT_OP(NFeatures::value,==,d);
            // 11 features
            auto mean = extract_result< tag::mean >(acc_);
            rBegin[0]  = mean;                                                             
            rBegin[1]  = mean*d;                                               
            rBegin[2]  = extract_result< tag::min >(acc_);                                 
            rBegin[3]  = extract_result< tag::max >(acc_);                                 
            rBegin[4]  = replaceRotten(extract_result< tag::moment<2> >(acc_),0.0);        
            rBegin[5]  = replaceRotten(extract_result< tag::moment<3> >(acc_),0.0);        
            rBegin[6]  = replaceRotten(quantile(acc_, quantile_probability = 0.1 ), mean);  
            rBegin[7]  = replaceRotten(quantile(acc_, quantile_probability = 0.25 ),mean); 
            rBegin[8]  = replaceRotten(quantile(acc_, quantile_probability = 0.5 ), mean);  
            rBegin[9]  = replaceRotten(quantile(acc_, quantile_probability = 0.75 ),mean); 
            rBegin[10] = replaceRotten(quantile(acc_, quantile_probability = 0.90 ),mean); 

        }

        size_t requiredPasses()const{
            return 1;
        }
        size_t nFeatures()const{
            return NFeatures::value;
        }
    private:

        T replaceRotten(const T & val, const T & replaceVal){
            if(std::isfinite(val))
                return val;
            else
                return replaceVal;
        }


        AccType acc_;


    };



}
}

#endif /*NIFTY_FEATURES_ACCUMULATED_FEATURES_HXX*/
