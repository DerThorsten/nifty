#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_CHUNKED_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_CHUNKED_HXX

#include "nifty/graph/rag/grid_rag_chunked.hxx"
#include "nifty/graph/rag/grid_rag_features.hxx"

namespace nifty{
namespace graph{
    
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
        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto & labels = labelsProxy.labels();

        // check that the data covers a whole slice in xy
        // need to take care of different axis ordering...
        NIFTY_CHECK_OP(data.shape(0),==,shape[2], "Shape along x does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(z0+data.shape(2),<=,shape[0], "Z offset is too large")


        vigra::Shape3 slice_shape(1,shape[1],shape[2]);
        vigra::MultiArray<3,LABELS_TYPE> this_slice(slice_shape);
        vigra::MultiArray<3,LABELS_TYPE> next_slice(slice_shape);
        const auto numberOfPasses =  std::max(edgeMap.numberOfPasses(),nodeMap.numberOfPasses());
        for(size_t p=0; p<numberOfPasses; ++p){
            // start pass p
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
                        
                        // this is shape(2), because it is the shape of a marray !
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


    
    template<class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const ChunkedLabelsGridRagSliced<LABELS_TYPE> & graph,
        const vigra::ChunkedArray<3,LABELS> & data, // template for the chunked data (expected to be a chunked vigra type)
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, 2> Coord;

        const auto & labelsProxy = graph.labelsProxy();
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
            
            nifty::tools::forEachCoordinate(std::array<int64_t,2>({(int64_t)x_max,(int64_t)y_max}),[&](const Coord & coord){
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


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_CHUNKED_HXX */
