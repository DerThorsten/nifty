#pragma once

#include <string>
#include <nifty/histogram/histogram.hxx>
#include <cmath>

#include "nifty/tools/runtime_check.hxx"
#include <nifty/nifty.hxx>

namespace nifty{
namespace graph{
namespace agglo{
namespace merge_rules{



    struct ArithmeticMeanSettings{
        auto name()const{
            return std::string("ArithmeticMean");
        }
    };

    template<class G, class T>
    class ArithmeticMeanEdgeMap{
    public:

        static auto staticName(){
            return std::string("ArithmeticMean");
        }
        auto name()const{
            return staticName();
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
            const auto ovalue  =  values_[deadEdge];
            const auto oweight = weights_[deadEdge];

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

        T weight(const uint64_t edge)const{
            return weights_[edge];
        }
    private:
        MeanEdgeMapType values_;
        SizeEdgeMapType weights_;
    };


    struct SumSettings{
        auto name()const{
            return std::string("Sum");
        }
    };

    template<class G, class T>
    class SumEdgeMap{
    public:

        static auto staticName(){
            return std::string("Sum");
        }
        auto name()const{
            return staticName();
        }

        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> SumEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T> SizeEdgeMapType;

        typedef SumSettings SettingsType;

        template<class VALUES, class WEIGHTS>
        SumEdgeMap(
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
            const auto ovalue  =  values_[deadEdge];
            const auto oweight = weights_[deadEdge];
            value += ovalue;
            weight += oweight;
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

        T weight(const uint64_t edge)const{
            return weights_[edge];
        }

        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        SumEdgeMapType values_;
        SizeEdgeMapType weights_;
    };


    struct GeneralizedMeanSettings{
        GeneralizedMeanSettings(const double p = 1.0)
        : p(p){
        }


        auto name()const{
            return std::string("GeneralizedMean") + std::string("[q=") + std::to_string(p) + std::string("]");
        }
        double p;
    };

    template<class G, class T>
    class  GeneralizedMeanEdgeMap{
    public:

        static auto staticName(){
            return std::string("GeneralizedMean");
        }
        auto name()const{
            return staticName() + std::string("[p=]") + std::to_string(settings_.p);
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
                    value = std::max(value, ovalue);
                }
                else{
                    value = std::min(value, ovalue);
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

        T weight(const uint64_t edge)const{
            return weights_[edge];
        }

        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MeanEdgeMapType values_;
        SizeEdgeMapType weights_;
        SettingsType settings_;
    };


    struct SmoothMaxSettings{
        double p;
        SmoothMaxSettings(double p = 1.0) : p(p){ }
        auto name()const{
            return std::string("SmoothMax") + std::string("[q=") + std::to_string(p) + std::string("]");
        }
    };

    template<class G, class T>
    class  SmoothMaxEdgeMap{
    public:

        static auto staticName(){
            return std::string("SmoothMax");
        }
        auto name()const{
            return staticName() + std::string("[q=") + std::to_string(settings_.p) + std::string("]");
        }
        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MeanEdgeMapType;
        typedef typename GraphType:: template EdgeMap<T> SizeEdgeMapType;

        typedef SmoothMaxSettings SettingsType;

        template<class VALUES, class WEIGHTS>
         SmoothMaxEdgeMap(
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
                    value = std::max(value, ovalue);
                }
                else{
                    value = std::min(value, ovalue);
                }
            }
            else if((p > 0.0-eps) && (p < 0.0 + eps)){
                value *= weight;
                value += oweight*ovalue;
                weight += oweight;
                value /= weight;
            }
            else{
                const auto sa = (weight ) * std::exp(value*p);
                const auto sd = (oweight) * std::exp(ovalue*p);

                value =  (value*sa + ovalue*sd)/(sa+sd);
                weight = weight + oweight;
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

        T weight(const uint64_t edge)const{
            return weights_[edge];
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
        double q;
        uint16_t numberOfBins;
        RankOrderSettings(double q = 0.5, uint16_t numberOfBins = 50) : q(q), numberOfBins(numberOfBins) { }
        auto name()const{
            std::stringstream ss;
            ss<<"RankOrderEdgeMap [q="<<q<<" #bins="<<numberOfBins<<"]";
            return ss.str();
        }
    };

    template<class G, class T>
    class  RankOrderEdgeMap{
    public:
        static auto staticName(){
            return std::string("RankOrderEdgeMap");
        }
        auto name()const{
            std::stringstream ss;
            ss<<staticName()<<" [q="<<settings_.q<<" #bins="<<settings_.numberOfBins<<"]";
            return ss.str();
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

        T weight(const uint64_t edge)const{
            NIFTY_CHECK(false,"Not implemented");
            return histogram_[edge].rank(settings_.q);
        }

        T operator[](const uint64_t edge)const{
            return histogram_[edge].rank(settings_.q);
        }
    private:
        HistogramEdgeMapType histogram_;
        SettingsType settings_;
    };


    struct MaxSettings{
        auto name()const{
            return std::string("Max");
        }
    };
    template<class G, class T>
    class MaxEdgeMap{
    public:


        static auto staticName(){
            return std::string("Max");
        }
        auto name()const{
            return staticName();
        }
        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MaxEdgeMapType;

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
        T weight(const uint64_t edge)const{
            return 1.0;
        }

        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MaxEdgeMapType values_;
    };

    struct MutexWatershedSettings{
        auto name()const{
            return std::string("MutexWatershed");
        }
    };
    template<class G, class T>
    class MutexWatershedEdgeMap{
    public:


        static auto staticName(){
            return std::string("MutexWatershed");
        }
        auto name()const{
            return staticName();
        }
        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MutexWatershedEdgeMapType;

        typedef MutexWatershedSettings SettingsType;

        template<class VALUES, class WEIGHTS>
        MutexWatershedEdgeMap(
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
            auto & aliveValue = values_[aliveEdge];
            auto const deadValue = values_[deadEdge];
            aliveValue = (std::abs(aliveValue) > std::abs(deadValue)) ? aliveValue : deadValue;
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
        T weight(const uint64_t edge)const{
            return 1.0;
        }

        T operator[](const uint64_t edge)const{
            return values_[edge];
        }
    private:
        MutexWatershedEdgeMapType values_;
    };


    struct MinSettings{
        auto name()const{
            return std::string("Min");
        }
    };

    template<class G, class T>
    class MinEdgeMap{
    public:


        static auto staticName(){
            return std::string("Min");
        }
        auto name()const{
            return staticName();
        }
        typedef G GraphType;
        typedef typename GraphType:: template EdgeMap<T> MinEdgeMapType;

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

        T weight(const uint64_t edge)const{
            return 1.0;
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

