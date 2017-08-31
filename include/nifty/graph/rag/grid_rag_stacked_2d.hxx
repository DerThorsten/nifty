#pragma once


#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag.hxx"

namespace nifty{
namespace graph{




template<class LABEL_PROXY>
class GridRagStacked2D
: public GridRag<3, LABEL_PROXY >
{
    typedef LABEL_PROXY LabelsProxyType;
    typedef GridRag<3, LABEL_PROXY > BaseType;
    typedef GridRagStacked2D< LABEL_PROXY > SelfType;
    typedef typename LabelsProxyType::LabelType LabelType;
    friend class detail_rag::ComputeRag< SelfType >;

    struct PerSliceData{
        PerSliceData(const LabelType numberOfLabels)
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
        LabelType minInSliceNode;
        LabelType maxInSliceNode;
    };
    
public:
    typedef typename BaseType::LabelsProxy LabelsProxy;
    typedef typename BaseType::SettingsType SettingsType;
    
    GridRagStacked2D(const LabelsProxy & labelsProxy, const SettingsType & settings = SettingsType())
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

    using BaseType::numberOfNodes;

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

