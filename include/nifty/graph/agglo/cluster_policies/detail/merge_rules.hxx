#pragma once

#include <string>
#include <nifty/histogram/histogram.hxx>

namespace nifty{
namespace graph{
namespace agglo{
namespace merge_rules{



    struct ArithmeticMeanSettings{};
    template<class G, class T>
    class ArithmeticMeanEdgeMap{
    public:

        static auto name(){
            return std::string("ArithmeticMean");
        }

        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MeanEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T> SizeEdgeMapType;

        typedef ArithmeticMeanSettings SettingsType;

        template<class VALUES, class WEIGHTS>
        ArithmeticMeanEdgeMap(
            const GraphType & g,
            const VALUES & values, 
            const WEIGHTS & weights,
            const SettingsType & settings = SettingsType()
        ):  values_(g),
            weights_(g)
        {
            for(auto edge : g.edges()){

                values_[edge] = values[edge];
                weights_[edge] = weights[edge];
            }
        }

        void merge(const uint64_t aliveEdge, const uint64_t deadEdge){

            auto & value  = values_[aliveEdge];
            auto & weight = weights_[aliveEdge];
            const auto & ovalue  =  values_[deadEdge];
            const auto & oweight = weights_[deadEdge];

            value *= weight;
            value += oweight*ovalue;
            weight += oweight;
            value /= weight;

        }

        void setValueFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
        }
        void setFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
            weights_[targetEdge] = weights_[sourceEdge];
        }

        void set(const uint64_t targetEdge, const T & value, const T &  weight){
            values_[targetEdge] = value;
            weights_[targetEdge] = weight;
        }
        
        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MeanEdgeMapType values_;
        SizeEdgeMapType weights_;
    };



    struct GeneralizedMeanSettings{
        double p = {1.0};
    };
    template<class G, class T>
    class  GeneralizedMeanEdgeMap{
    public:

        static auto name(){
            return std::string("GeneralizedMean");
        }

        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MeanEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T> SizeEdgeMapType;

        typedef GeneralizedMeanSettings SettingsType;

        template<class VALUES, class WEIGHTS>
         GeneralizedMeanEdgeMap(
            const GraphType & g,
            const VALUES & values, 
            const WEIGHTS & weights,
            const SettingsType & settings = SettingsType()
        ):  values_(g),
            weights_(g),
            settings_(settings)
        {
            for(auto edge : g.edges()){

                values_[edge] = values[edge];
                weights_[edge] = weights[edge];
            }
        }

        void merge(const uint64_t aliveEdge, const uint64_t deadEdge){

            auto & value  = values_[aliveEdge];
            auto & weight = weights_[aliveEdge];
            const auto & ovalue  =  values_[deadEdge];
            const auto & oweight = weights_[deadEdge];

            const auto p = settings_.p;
            const static auto eps = 0.0000001;

            if(std::isinf(p)){
                if(p>0){
                    weight = std::max(weight, oweight);
                }
                else{
                    weight = std::min(weight, oweight);
                }
            }
            else if((p > 1.0-eps) && (p < 1.0 + eps)){
                value *= weight;
                value += oweight*ovalue;
                weight += oweight;
                value /= weight;
            }
            else{
                const auto s = weight + oweight;
                const auto sa = (weight/s ) * std::pow(value, p);
                const auto sd = (oweight/s) * std::pow(ovalue, p);
                value = std::pow(sa+sd, 1.0/p);
                weight = s;
            }
        }

        void setValueFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
        }
        void setFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
            weights_[targetEdge] = weights_[sourceEdge];
        }

        void set(const uint64_t targetEdge, const T & value, const T &  weight){
            values_[targetEdge] = value;
            weights_[targetEdge] = weight;
        }
        
        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MeanEdgeMapType values_;
        SizeEdgeMapType weights_;
        SettingsType settings_;
    };


    struct RankOrderSettings{
        double q = {0.5};
        uint16_t numberOfBins = {50};
    };
    template<class G, class T>
    class  RankOrderEdgeMap{
    public:
        static auto name(){
            return std::string("RankOrderEdgeMap");
        }

        typedef G GraphType;
        typedef nifty::histogram::Histogram<double, double>          HistogramType;
        typedef typename GraphType:: template EdgeMap<HistogramType> HistogramEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T>             SizeEdgeMapType;

        typedef RankOrderSettings SettingsType;

        template<class VALUES, class WEIGHTS>
         RankOrderEdgeMap(
            const GraphType & g,
            const VALUES & values, 
            const WEIGHTS & weights,
            const SettingsType & settings = SettingsType()
        ):  histogram_(g),
            settings_(settings)
        {
            T minVal =        std::numeric_limits<T>::infinity();
            T maxVal = -1.0 * std::numeric_limits<T>::infinity();

            for(auto edge : g.edges()){

                const auto val = T(values[edge]);
                maxVal = std::max(maxVal, val);
                minVal = std::min(minVal, val);
            }

            for(auto edge : g.edges()){
                auto & hist = histogram_[edge];
                hist.assign(minVal, maxVal, settings_.numberOfBins);
                hist.insert(values[edge], weights[edge]);
            }
        }

        void merge(const uint64_t aliveEdge, const uint64_t deadEdge){
            histogram_[aliveEdge].merge(histogram_[deadEdge]);
        }

        void setValueFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            auto & thist= histogram_[targetEdge];
            const auto &  shist = histogram_[sourceEdge];

            const auto tsum = thist.sum();
            thist = shist;
            thist.normalize(tsum);

        }
        void setFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            histogram_[targetEdge] = histogram_[sourceEdge];
        }
        void set(const uint64_t targetEdge, const T & value, const T &  weight){
            auto & hist =  histogram_[targetEdge];
            hist.clearCounts();
            hist.insert(value, weight);
        }
        
        T operator[](const uint64_t edge)const{
            return histogram_[edge].rank(settings_.q);
        }
    private:
        HistogramEdgeMapType histogram_;
        SettingsType settings_;
    };


    struct MaxSettings{};
    template<class G, class T>
    class MaxEdgeMap{
    public:


        static auto name(){
            return std::string("Max");
        }

        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MaxEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T> SizeEdgeMapType;

        typedef MaxSettings SettingsType;

        template<class VALUES, class WEIGHTS>
        MaxEdgeMap(
            const GraphType & g, 
            const VALUES & values, 
            const WEIGHTS & weights,
            const SettingsType & settings = SettingsType()
        ):  values_(g)
        {
            for(auto edge : g.edges()){
                values_[edge] = values[edge];
            }
        }

        void merge(const uint64_t aliveEdge, const uint64_t deadEdge){
            auto & value = values_[aliveEdge];
            value = std::max(value, values_[deadEdge]);
        }

        void setValueFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
        }
        void setFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
        }
        void set(const uint64_t targetEdge, const T & value, const T &  weight){
            values_[targetEdge] = value;
        }
        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MaxEdgeMapType values_;
    };

    struct MinSettings{};
    template<class G, class T>
    class MinEdgeMap{
    public:


        static auto name(){
            return std::string("Min");
        }

        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MinEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T> SizeEdgeMapType;

        typedef MinSettings SettingsType;

        template<class VALUES, class WEIGHTS>
        MinEdgeMap(
            const GraphType & g, 
            const VALUES & values, 
            const WEIGHTS & weights,
            const SettingsType & settings = SettingsType()
        ):  values_(g)
        {
            for(auto edge : g.edges()){
                values_[edge] = values[edge];
            }
        }

        void merge(const uint64_t aliveEdge, const uint64_t deadEdge){
            auto & value = values_[aliveEdge];
            value = std::min(value, values_[deadEdge]);
        }

        void setValueFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
        }
        void setFrom(const uint64_t targetEdge, const uint64_t sourceEdge){
            values_[targetEdge] = values_[sourceEdge];
        }
        void set(const uint64_t targetEdge, const T & value, const T &  weight){
            values_[targetEdge] = value;
        }
        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MinEdgeMapType values_;
    };




} // merge rule
} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

