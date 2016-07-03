#pragma once
#ifndef NIFTY_GRAPH_GALA_FEATURE_BASE_HXX
#define NIFTY_GRAPH_GALA_FEATURE_BASE_HXX

#include <vector>
#include <memory>

namespace nifty{
namespace graph{

    
    

    template<class GRAPH, class T>
    class GalaFeatureBase{
    public:
        virtual ~GalaFeatureBase(){
        }
        virtual uint64_t numberOfFeatures() const = 0;
        virtual void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge) = 0;
        virtual void mergeNodes(const uint64_t aliveEdge, const uint64_t deadEdge) = 0;
        virtual void getFeatures(const uint64_t edge, T * featuresOut) = 0;
        virtual void reset() =0;
    };

    template<class GRAPH, class T>
    class DummyFeature{
    public:
        virtual ~DummyFeature(){
        }
        virtual uint64_t numberOfFeatures() const {
           return  2;
        }
        virtual void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge) {

        }
        virtual void mergeNodes(const uint64_t aliveEdge, const uint64_t deadEdge) {

        }
        virtual void getFeatures(const uint64_t edge, T * featuresOut) {
            featuresOut[0] = 1.0;
            featuresOut[1] = 2.0;
        }
        virtual void reset() {

        }
    };

    template<class GRAPH, class T>
    class GalaFeatureCollection : GalaFeatureBase<GRAPH, T>{
    public:
        typedef GRAPH GraphType;
        typedef GalaFeatureBase<GraphType, T>       FeatureBaseType;
        typedef std::shared_ptr<FeatureBaseType>    FeatureBaseTypeSharedPtr;

        virtual uint64_t numberOfFeatures() const {
            auto nf = 0;
            for(auto f : featuresRawPtr_)
                nf += f->numberOfFeatures();
            return nf;
        }
        virtual void mergeEdges(const uint64_t alive, const uint64_t dead) {
            for(auto f : featuresRawPtr_)
               f->mergeEdges(alive, dead);
        }
        virtual void mergeNodes(const uint64_t alive, const uint64_t dead) {
            auto nf = 0;
            for(auto f : featuresRawPtr_)
               f->mergeNodes(alive, dead);
        }
        virtual void getFeatures(const uint64_t edge, T * featuresOut){
            auto fout = featuresOut;
            auto nf = 0;
            for(auto f : featuresRawPtr_){
                auto nf =  f->numberOfFeatures();
                f->getFeatures(edge, fout);
                fout += nf;
            }
        }
        virtual void reset(){
            for(auto f : featuresRawPtr_){
                f->reset();
            }
        }
    private:
        std::vector<FeatureBaseTypeSharedPtr> featuresSharedPtr_;
        std::vector<FeatureBaseType *>        featuresRawPtr_;
    };




} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_FEATURE_BASE_HXX
