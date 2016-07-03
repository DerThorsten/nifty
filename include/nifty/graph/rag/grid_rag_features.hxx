#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"


namespace nifty{
namespace graph{

    template<class T, unsigned int NBINS>
    class DefaultAcc{
        
    public:
        DefaultAcc()
        :   minVal_(std::numeric_limits<T>::infinity()),
            maxVal_(-1.0*std::numeric_limits<T>::infinity()),
            sumVal_(0.0),
            countVal_(0.0)
        {

        }
        void accumulate(const T & val){
            minVal_ = std::min(val, minVal_);
            maxVal_ = std::max(val, maxVal_);
            sumVal_ += val;
            ++countVal_;
        }
    private:
        T minVal_,maxVal_,sumVal_;
        uint64_t countVal_;
    };


    template<class GRAPH, class T, unsigned int NBINS, template<class> class ITEM_MAP >
    class DefaultAccMapBase 
    {
    public:
        typedef GRAPH Graph;
        DefaultAccMapBase(const Graph & graph, const T globalMinVal, const T globalMaxVal)
        :   currentPass_(0),
            accs_(graph){

        }
        void accumulate(const uint64_t item, const T & val){
            if(currentPass_ == 0){
                accs_[item].accumulate(val);
            }
        }
        void startPass(const size_t passIndex){
            currentPass_ = passIndex;
        }
        size_t numberOfPasses()const{
            return 1.0;
        }
    private:
        size_t currentPass_;
        ITEM_MAP<DefaultAcc<T, NBINS> > accs_;
    };



    template<class GRAPH, class T, unsigned int NBINS=40>
    class DefaultAccEdgeMap : 
        public  DefaultAccMapBase<GRAPH, T, NBINS, GRAPH:: template EdgeMap >
    {
    public:
        typedef GRAPH Graph;
        //typedef typename Graph:: template EdgeMap<DefaultAcc<T, NBINS> > BasesBaseType;
        typedef DefaultAccMapBase<Graph, T, NBINS, Graph:: template EdgeMap >  BaseType;
        using BaseType::BaseType;
    };

    template<class GRAPH, class T, unsigned int NBINS=40>
    class DefaultAccNodeMap : 
        public  DefaultAccMapBase<GRAPH, T, NBINS, GRAPH:: template NodeMap >
    {
    public:
        typedef GRAPH Graph;
        //typedef typename Graph:: template NodeMap<DefaultAcc<T, NBINS> > BasesBaseType;
        typedef DefaultAccMapBase<Graph, T, NBINS, Graph:: template NodeMap >  BaseType;
        using BaseType::BaseType;
    };



    

    

    template<class LABELS_TYPE, class T, class EDGE_MAP, class NODE_MAP>
    void gridRagAccumulateFeatures(
        const ExplicitLabelsGridRag<2, LABELS_TYPE> & graph,
        nifty::marray::View<T> data,
        EDGE_MAP & edgeMap,
        NODE_MAP &  nodeMap
    ){
        const auto labelsProxy = graph.labelsProxy();
        const auto labels = labelsProxy.labels(); 
        
        const auto numberOfPasses =  std::max(edgeMap.numberOfPasses(),nodeMap.numberOfPasses());
        for(size_t p=0; p<numberOfPasses; ++p){
            // start path p
            edgeMap.startPass(p);
            nodeMap.startPass(p);
            for(size_t x=0; x<labels.shape(0); ++x)
            for(size_t y=0; y<labels.shape(1); ++y){

                const auto lU = labels(x, y);
                const auto dU = data(x,y);

                nodeMap.accumulate(lU, dU);

                if(x+1<labels.shape(0)){
                    const auto lV = labels(x+1, y);
                    if(lU != lV){
                        const auto e = graph.findEdge(lU, lV);
                        const auto dV = data(x+1,y);
                        edgeMap.accumulate(e, dU);
                        edgeMap.accumulate(e, dV);
                    }
                }
                if(y+1<labels.shape(1)){
                    const auto lV = labels(x, y+1);
                    if(lU != lV){
                        const auto e = graph.findEdge(lU, lV);
                        const auto dV = data(x,y+1);
                        edgeMap.accumulate(e, dU);
                        edgeMap.accumulate(e, dV);
                    }
                }
            }
        }
    }

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX */
