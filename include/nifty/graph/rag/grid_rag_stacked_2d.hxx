#pragma once

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag.hxx"

#include "nifty/graph/rag/detail_rag/compute_grid_rag_stacked.hxx"


namespace nifty{
namespace graph{


template<class LABELS>
class GridRagStacked2D : public GridRag<3, LABELS> {

    typedef GridRag<3, LABELS > BaseType;
    typedef GridRagStacked2D<LABELS> SelfType;
    typedef typename LABELS::value_type value_type;
    friend class detail_rag::ComputeRag<SelfType>;

    struct PerSliceData{
        PerSliceData(const value_type numberOfLabels)
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
        value_type minInSliceNode;
        value_type maxInSliceNode;
    };

public:
    typedef typename BaseType::LabelsType LabelsType;
    typedef typename BaseType::SettingsType SettingsType;
    typedef typename BaseType::BlockStorageType BlockStorageType;

    GridRagStacked2D(const LabelsType & labels,
                     const std::size_t numberOfLabels,
                     const SettingsType & settings = SettingsType())
    :   BaseType(labels, numberOfLabels, settings, typename BaseType::DontComputeRag()),
        perSliceDataVec_(
            labels.shape()[0],
            PerSliceData(numberOfLabels)
        ),
        numberOfInSliceEdges_(0),
        numberOfInBetweenSliceEdges_(0),
        edgeLengths_()
    {
        detail_rag::ComputeRag< SelfType >::computeRag(*this, this->settings_);
    }

    template<class ITER>
    GridRagStacked2D(const LabelsType & labels,
                     const std::size_t numberOfLabels,
                     ITER serializationBegin,
                     const SettingsType & settings = SettingsType())
    :   BaseType(labels, numberOfLabels, settings, typename BaseType::DontComputeRag()),
        perSliceDataVec_(
            labels.shape()[0],
            PerSliceData(numberOfLabels)
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
    const std::vector<std::size_t> & edgeLengths() const {
        return edgeLengths_;
    }

    // additional serialisation
    uint64_t serializationSize() const;

    template<class ITER>
    void serialize(ITER & iter) const;

    template<class ITER>
    void deserialize(ITER & iter);

    // ignore label api
    bool haveIgnoreLabel() const
    {return BaseType::settings_.haveIgnoreLabel;}

    uint64_t ignoreLabel() const
    {return BaseType::settings_.ignoreLabel;}
private:

    std::vector<PerSliceData> perSliceDataVec_;
    uint64_t numberOfInSliceEdges_;
    uint64_t numberOfInBetweenSliceEdges_;
    std::vector<std::size_t> edgeLengths_;
};

template<class LABELS>
uint64_t GridRagStacked2D<LABELS>::serializationSize() const {
    return BaseType::serializationSize() + perSliceDataVec_.size() * 6 + 2 + this->numberOfEdges();
}

template<class LABELS>
template<class ITER>
void GridRagStacked2D<LABELS>::serialize(ITER & iter) const {

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

template<class LABELS>
template<class ITER>
void GridRagStacked2D<LABELS>::deserialize(ITER & iter) {

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
