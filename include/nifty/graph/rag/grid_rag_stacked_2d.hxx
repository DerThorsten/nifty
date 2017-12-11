#pragma once

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag.hxx"

#include "nifty/graph/rag/detail_rag/compute_grid_rag_stacked.hxx"


namespace nifty{
namespace graph{


template<class LABEL_PROXY>
class GridRagStacked2D : public GridRag<3, LABEL_PROXY> {

    typedef GridRag<3, LABEL_PROXY > BaseType;
    typedef GridRagStacked2D< LABEL_PROXY > SelfType;
    typedef typename LABEL_PROXY::LabelType LabelType;
    friend class detail_rag::ComputeRag<SelfType>;

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
        numberOfInBetweenSliceEdges_(0),
        edgeLengths_()
    {
        detail_rag::ComputeRag< SelfType >::computeRag(*this, this->settings_);
    }

    template<class ITER>
    GridRagStacked2D(const LabelsProxy & labelsProxy,
            ITER serializationBegin,
            const SettingsType & settings = SettingsType())
    :   BaseType(labelsProxy, settings, typename BaseType::DontComputeRag()),
        perSliceDataVec_(
            labelsProxy.shape()[0],
            PerSliceData(labelsProxy.numberOfLabels())
        ),
        numberOfInSliceEdges_(0),
        numberOfInBetweenSliceEdges_(0),
        edgeLengths_()
    {
        this->deserialize(serializationBegin);
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
    uint64_t numberOfInSliceEdges() const {
        return numberOfInSliceEdges_;
    }
    uint64_t numberOfInBetweenSliceEdges() const {
        return numberOfInBetweenSliceEdges_;
    }
    const std::vector<size_t> & edgeLengths() const {
        return edgeLengths_;
    }

    // additional serialisation
    uint64_t serializationSize() const;

    template<class ITER>
    void serialize(ITER & iter) const;

    template<class ITER>
    void deserialize(ITER & iter);
private:

    std::vector<PerSliceData> perSliceDataVec_;
    uint64_t numberOfInSliceEdges_;
    uint64_t numberOfInBetweenSliceEdges_;
    std::vector<size_t> edgeLengths_;
};

template<class LABEL_PROXY>
uint64_t GridRagStacked2D<LABEL_PROXY>::serializationSize() const {
    return BaseType::serializationSize() + perSliceDataVec_.size() * 6 + 2 + this->numberOfEdges();
}

template<class LABEL_PROXY>
template<class ITER>
void GridRagStacked2D<LABEL_PROXY>::serialize(ITER & iter) const {

    BaseType::serialize(iter);
    *iter = this->numberOfInSliceEdges_;
    ++iter;
    *iter = this->numberOfInBetweenSliceEdges_;
    ++iter;
    for(const auto & perSliceData : this->perSliceDataVec_ ) {
        *iter = perSliceData.numberOfInSliceEdges;
        ++iter;
    }
    for(const auto & perSliceData : this->perSliceDataVec_ ) {
        *iter = perSliceData.numberOfToNextSliceEdges;
        ++iter;
    }
    for(const auto & perSliceData : this->perSliceDataVec_ ) {
        *iter = perSliceData.inSliceEdgeOffset;
        ++iter;
    }
    for(const auto & perSliceData : this->perSliceDataVec_ ) {
        *iter = perSliceData.toNextSliceEdgeOffset;
        ++iter;
    }
    for(const auto & perSliceData : this->perSliceDataVec_ ) {
        *iter = perSliceData.minInSliceNode;
        ++iter;
    }
    for(const auto & perSliceData : this->perSliceDataVec_ ) {
        *iter = perSliceData.maxInSliceNode;
        ++iter;
    }
    for(const auto len : edgeLengths_) {
        *iter = len;
        ++iter;
    }
}

template<class LABEL_PROXY>
template<class ITER>
void GridRagStacked2D<LABEL_PROXY>::deserialize(ITER & iter) {

    BaseType::deserialize(iter);
    this->numberOfInSliceEdges_ = *iter;
    ++iter;
    this->numberOfInBetweenSliceEdges_ = *iter;
    ++iter;
    for(auto & perSliceData : this->perSliceDataVec_ ) {
        perSliceData.numberOfInSliceEdges = *iter;
        ++iter;
    }
    for(auto & perSliceData : this->perSliceDataVec_ ) {
        perSliceData.numberOfToNextSliceEdges = *iter;
        ++iter;
    }
    for(auto & perSliceData : this->perSliceDataVec_ ) {
        perSliceData.inSliceEdgeOffset = *iter;
        ++iter;
    }
    for(auto & perSliceData : this->perSliceDataVec_ ) {
        perSliceData.toNextSliceEdgeOffset = *iter;
        ++iter;
    }
    for(auto & perSliceData : this->perSliceDataVec_ ) {
        perSliceData.minInSliceNode = *iter;
        ++iter;
    }
    for(auto & perSliceData : this->perSliceDataVec_ ) {
        perSliceData.maxInSliceNode = *iter;
        ++iter;
    }
    edgeLengths_.resize(this->numberOfEdges());
    for(auto & len : edgeLengths_ ) {
        len = *iter;
        ++iter;
    }
}


} // end namespace graph
} // end namespace nifty
