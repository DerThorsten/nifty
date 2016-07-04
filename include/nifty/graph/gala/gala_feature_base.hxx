#pragma once
#ifndef NIFTY_GRAPH_GALA_FEATURE_BASE_HXX
#define NIFTY_GRAPH_GALA_FEATURE_BASE_HXX

#include <vector>
#include <memory>

#include "nifty/graph/rag/grid_rag_features.hxx"

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


    template<class GRAPH, class T, unsigned int NBINS = 40>
    class GalaDefaultAccFeature  : GalaFeatureBase<GRAPH, T>{

    public:
        typedef GRAPH GraphType;
        typedef DefaultAccNodeMap<GRAPH, T, NBINS> NodeMap;
        typedef DefaultAccEdgeMap<GRAPH, T, NBINS> EdgeMap;

        GalaDefaultAccFeature(const GRAPH & graph, const EdgeMap & edgeMap, const NodeMap & nodeMap)
        :   graph_(graph),
            edgeMapIn_(edgeMap),
            nodeMapIn_(nodeMap),
            edgeMap_(graph),
            nodeMap_(graph){

            // clone content
            this->reset();
        }
        virtual uint64_t numberOfFeatures() const {
            return edgeMapIn_.numberOfFeatures() + 4*nodeMapIn_.numberOfFeatures();
        }
        virtual void mergeEdges(const uint64_t alive, const uint64_t dead) {
            edgeMap_.merge(alive, dead);
        }
        virtual void mergeNodes(const uint64_t alive, const uint64_t dead) {
            nodeMap_.merge(alive, dead);
        }
        virtual void getFeatures(const uint64_t edge, T * featuresOut) {

            T * fOut = featuresOut;
            edgeMap_.getFeatures(edge, fOut);
            fOut += edgeMapIn_.numberOfFeatures();

            const auto uv = graph_.uv(edge);
            std::vector<T> uFeat(nodeMapIn_.numberOfFeatures());
            std::vector<T> vFeat(nodeMapIn_.numberOfFeatures());
            nodeMap_.getFeatures(uv.first, uFeat.data());
            nodeMap_.getFeatures(uv.second, uFeat.data());

            for(size_t i=0; i<uFeat.size(); ++i){
                fOut[i*4 + 0] = std::min(uFeat[i],vFeat[i]);
                fOut[i*4 + 1] = std::max(uFeat[i],vFeat[i]);
                fOut[i*4 + 1] = std::abs(uFeat[i]-vFeat[i]);
                fOut[i*4 + 1] = uFeat[i]+vFeat[i];
            }
        }
        virtual void reset() {
            edgeMap_.resetFrom(edgeMapIn_);
            nodeMap_.resetFrom(nodeMapIn_);
        }

    private:
        const GraphType & graph_;
        const EdgeMap & edgeMapIn_;
        const NodeMap & nodeMapIn_;
        EdgeMap edgeMap_;
        NodeMap nodeMap_;
        

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
        //std::vector<FeatureBaseTypeSharedPtr> featuresSharedPtr_;
        std::vector<FeatureBaseType *>        featuresRawPtr_;
    };




} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_GALA_FEATURE_BASE_HXX
