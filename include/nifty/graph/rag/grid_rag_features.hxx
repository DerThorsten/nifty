#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"

#include "nifty/tools/for_each_coordinate.hxx"

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
        void merge(const DefaultAcc & other){
            minVal_ = std::min(other.minVal_, minVal_);
            maxVal_ = std::max(other.maxVal_, maxVal_);
            sumVal_ += other.sumVal_;
            countVal_ +=other.countVal_;
        }
        void getFeatures(T * featuresOut){
            featuresOut[0] = minVal_;
            featuresOut[1] = maxVal_;
            featuresOut[2] = sumVal_;
            featuresOut[3] = sumVal_/static_cast<T>(countVal_);
            featuresOut[4] = static_cast<T>(countVal_);
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
        DefaultAccMapBase(const Graph & graph, const T globalMinVal = std::numeric_limits<T>::infinity(), const T globalMaxVal=-1.0*std::numeric_limits<T>::infinity())
        :   graph_(graph),
            currentPass_(0),
            globalMinVal_(globalMinVal),
            globalMaxVal_(globalMaxVal),
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
            return 1;
        }
        void merge(const uint64_t alive,  const uint64_t dead){
            accs_[alive].merge(accs_[dead]);
        }
        uint64_t numberOfFeatures() const {
            return 5;
        }
        void getFeatures(const uint64_t item, T * featuresOut){
            accs_[item].getFeatures(featuresOut);
        }   
        void resetFrom(const DefaultAccMapBase & other){
            std::copy(other.accs_.begin(), other.accs_.end(), accs_.begin());
            globalMinVal_ = other.globalMinVal_;
            globalMinVal_ = other.globalMaxVal_;
            currentPass_ = other.currentPass_;
        }
    private:
        const GRAPH & graph_;
        size_t currentPass_;
        T globalMinVal_;
        T globalMaxVal_;
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



    

    

    template< size_t DIM, class LABELS_TYPE, class T, class EDGE_MAP, class NODE_MAP>
    void gridRagAccumulateFeatures(
        const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
        nifty::marray::View<T> data,
        EDGE_MAP & edgeMap,
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, DIM> Coord;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels(); 
        
        const auto numberOfPasses =  std::max(edgeMap.numberOfPasses(),nodeMap.numberOfPasses());
        for(size_t p=0; p<numberOfPasses; ++p){
            // start path p
            edgeMap.startPass(p);
            nodeMap.startPass(p);

            nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
                const auto lU = labels(coord);
                const auto dU = data(coord);
                nodeMap.accumulate(lU, dU);
                for(size_t axis=0; axis<DIM; ++axis){
                    Coord coord2 = coord;
                    coord2[axis] += 1;
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2);
                        if(lU != lV){
                            const auto e = graph.findEdge(lU, lV);
                            const auto dV = data(coord2);
                            edgeMap.accumulate(e, dU);
                            edgeMap.accumulate(e, dV);
                        }
                    }
                }
            });
        }
    }
    
    
    template< class LABELS_TYPE, class T, class EDGE_MAP, class NODE_MAP>
    void gridRagAccumulateFeatures(
        const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
        // FIXME have to extremly cautious when giving this data as mArray, because of different axis order than vigra...
        // maybe use vigra multi arrays instead ?!
        const  marray::View<T> & data,
        EDGE_MAP & edgeMap,
        NODE_MAP &  nodeMap,
        const size_t z0
    ){
        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels();

        // check that the data covers a whole slice in xy
        // need to take care of different axis ordering...
        NIFTY_CHECK_OP(data.shape(0),==,shape[2], "Shape along x does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(z0+data.shape(2),<,shape[0], "Z offset is too large")


        vigra::Shape3 slice_shape(1,shape[1],shape[2]);
        vigra::MultiArray<3,LABELS_TYPE> this_slice(slice_shape);
        vigra::MultiArray<3,LABELS_TYPE> next_slice(slice_shape);
        const auto numberOfPasses =  std::max(edgeMap.numberOfPasses(),nodeMap.numberOfPasses());
        for(size_t p=0; p<numberOfPasses; ++p){
            // start path p
            edgeMap.startPass(p);
            nodeMap.startPass(p);

            for( size_t z = 0; z < data.shape(2); z++ )
            {
                vigra::Shape3 this_begin(z+z0,0,0);
                labels.checkoutSubarray(this_begin, this_slice);

                if( z < data.shape(2) - 1) {
                    vigra::Shape3 next_begin(z+z0+1,0,0);
                    labels.checkoutSubarray(next_begin, next_slice);
                }
                
                // TODO parallelize
                for(size_t y = 0; y < shape[1]; y++) {
                    for(size_t x = 0; x < shape[2]; x++) {
                        
                        const auto lU = this_slice(0,y,x);
                        const auto dU = data(x,y,z);
                        nodeMap.accumulate(lU, dU);
                        
                        if( x + 1 < shape[2] ) {
                            const auto lV = this_slice(0,y,x+1);
                            const auto dV = data(x+1,y,z);
                            if( lU != lV) {
                                const auto e = graph.findEdge(lU, lV);
                                edgeMap.accumulate(e, dU);
                                edgeMap.accumulate(e, dV);
                            }
                        }
                        
                        if( y + 1 < shape[1] ) {
                            const auto lV = this_slice(0,y+1,x);
                            const auto dV = data(x,y+1,z);
                            if( lU != lV) {
                                const auto e = graph.findEdge(lU, lV);
                                edgeMap.accumulate(e, dU);
                                edgeMap.accumulate(e, dV);
                            }
                        }
                        
                        if( z + 1 < data.shape(2)) {
                            const auto lV = next_slice(0,y,x);
                            const auto dV = data(x,y,z+1);
                            if( lU != lV) {
                                const auto e = graph.findEdge(lU, lV);
                                edgeMap.accumulate(e, dU);
                                edgeMap.accumulate(e, dV);
                            }
                        }

                    }
                }

            }
        }
    }

    
    template<size_t DIM, class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
        nifty::marray::View<LABELS> data,
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, DIM> Coord;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto labels = labelsProxy.labels(); 

        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
        

        nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
            const auto x = coord[0];
            const auto y = coord[1];
            const auto node = labels(x,y);            
            const auto l  = data(x,y);
            overlaps[node][l] += 1;
        });

        for(const auto node : graph.nodes()){
            const auto & ol = overlaps[node];
            // find max ol 
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[node] = maxOlLabel;
        }
    }
    
    
    // TODO implement
    template<class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
        const vigra::ChunkedArray<3,LABELS> & data, // template for the chunked data (expected to be a chunked vigra type)
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, 2> Coord;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels(); 

        // check that the data covers a whole slice in xy
        // need to take care of different axis ordering...
        NIFTY_CHECK_OP(data.shape(2),==,shape[2], "Shape along x does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along z does not agree")

        size_t z_max = shape[0];
        size_t y_max = shape[1];
        size_t x_max = shape[2];

        vigra::Shape3 slice_shape(1, y_max, x_max);

        vigra::MultiArray<3,LABELS_TYPE> this_labels(slice_shape);
        vigra::MultiArray<3,LABELS> this_data(slice_shape);
        
        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());

        for(size_t z = 0; z < z_max; z++) {
            
            // checkout this slice
            vigra::Shape3 slice_begin(z, 0, 0);
            labels.checkoutSubarray(slice_begin, this_labels);
            data.checkoutSubarray(slice_begin, this_data);
            
            // TODO parallel versions of the code
            nifty::tools::forEachCoordinate(std::array<int64_t,2>({shape[x_max,y_max]}),[&](const Coord & coord){
                const auto x = coord[0];
                const auto y = coord[1];
                const auto node = this_labels(0,y,x);            
                const auto l  = this_data(0,y,x);
                overlaps[node][l] += 1;
            });
        }
        
        for(const auto node : graph.nodes()){
            const auto & ol = overlaps[node];
            // find max ol 
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[node] = maxOlLabel;
        }

    }
    

} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX */
