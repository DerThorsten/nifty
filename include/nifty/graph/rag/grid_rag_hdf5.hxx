#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag_hdf5.hxx"


namespace nifty{
namespace graph{


template<class LABEL_TYPE>
class GridRagStacked2D< Hdf5Labels<3, LABEL_TYPE> >
: public GridRag<3, Hdf5Labels<3, LABEL_TYPE> >
{

    typedef GridRag<3, Hdf5Labels<3, LABEL_TYPE> > BaseType;
    typedef GridRagStacked2D< Hdf5Labels<3, LABEL_TYPE> > SelfType;
    typedef LABEL_TYPE LabelType;
    friend class detail_rag::ComputeRag< SelfType >;

    struct PerSliceData{
        PerSliceData(const LABEL_TYPE numberOfLabels)
        :   numberOfInSliceEdges(0),
            numberOfToNextSliceEdges(0),
            inSliceEdgeOffset(0),
            toNextSliceEdgeOffset(0),
            minInSliceNode(numberOfLabels),
            maxInSliceNode(0){
        }
        uint64_t numberOfInSliceEdges;
        uint64_t numberOfToNextSliceEdges;
        uint64_t inSliceEdgeOffset;
        uint64_t toNextSliceEdgeOffset;
        LABEL_TYPE minInSliceNode;
        LABEL_TYPE maxInSliceNode;
    };
    
public:
    typedef typename BaseType::LabelsProxy LabelsProxy;
    typedef typename BaseType::Settings Settings;
    
    GridRagStacked2D(const LabelsProxy & labelsProxy, const Settings & settings = Settings())
    :   BaseType(labelsProxy, settings, typename BaseType::DontComputeRag()),
        perSliceDataVec_(
            labelsProxy.shape()[0], 
            PerSliceData(labelsProxy.numberOfLabels()) 
        ),
        numberOfInSliceEdges_(0),
        numberOfInBetweenSliceEdges_(0)
    {

        detail_rag::ComputeRag< SelfType >::computeRag(*this, this->settings_);
    }

    // additional api
    std::pair<uint64_t, uint64_t> minMaxNode(const uint64_t sliceIndex) const{
        const auto & sliceData = perSliceDataVec_[sliceIndex];
        return std::pair<uint64_t, uint64_t>(sliceData.minInSliceNode, sliceData.maxInSliceNode);
    }
    uint64_t numberOfNodes(const uint64_t sliceIndex) const{
        const auto & sliceData = perSliceDataVec_[sliceIndex];
        return (sliceData.maxInSliceNode-sliceData.minInSliceNode) + 1;
    }
    uint64_t numberOfInSliceEdges(const uint64_t sliceIndex) const{
        const auto & sliceData = perSliceDataVec_[sliceIndex];
        return sliceData.numberOfInSliceEdges;
    }
    uint64_t numberOfInBetweenSliceEdges(const uint64_t sliceIndex) const{
        const auto & sliceData = perSliceDataVec_[sliceIndex];
        return sliceData.numberOfToNextSliceEdges;
    }
    uint64_t inSliceEdgeOffset(const uint64_t sliceIndex) const{
        const auto & sliceData = perSliceDataVec_[sliceIndex];
        return sliceData.inSliceEdgeOffset;
    }
    uint64_t betweenSliceEdgeOffset(const uint64_t sliceIndex) const{
        const auto & sliceData = perSliceDataVec_[sliceIndex];
        return sliceData.toNextSliceEdgeOffset;
    }
private:

    std::vector<PerSliceData> perSliceDataVec_;
    uint64_t numberOfInSliceEdges_;
    uint64_t numberOfInBetweenSliceEdges_;
};






} // end namespace graph
} // end namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX */
