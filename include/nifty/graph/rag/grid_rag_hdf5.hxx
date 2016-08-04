#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag_hdf5.hxx"


namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class GridRag<DIM, Hdf5Labels<DIM, LABEL_TYPE> > : public UndirectedGraph<>{
public:

    typedef Hdf5Labels<DIM, LABEL_TYPE> LabelsProxy;
    struct Settings{
        Settings()
        :   numberOfThreads(-1),
            blockShape()
        {
            for(auto d=0; d<DIM; ++d)
                blockShape[d] = 100;
        }
        int numberOfThreads;
        array::StaticArray<int64_t, DIM> blockShape;
    };

    typedef GridRag<DIM, Hdf5Labels<DIM, LABEL_TYPE> > SelfType;

    friend class detail_rag::ComputeRag< SelfType >;


    GridRag(const LabelsProxy & labelsProxy, const Settings & settings = Settings())
    :   settings_(settings),
        labelsProxy_(labelsProxy)
    {
        detail_rag::ComputeRag< SelfType >::computeRag(*this, settings_);
    }

    const LabelsProxy & labelsProxy() const {
        return labelsProxy_;
    }
private:
    Settings settings_;
    const LabelsProxy & labelsProxy_;

};




} // end namespace graph
} // end namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX */
