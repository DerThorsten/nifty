#include "z5/factory.hxx"
#include "z5/util/util.hxx"
#include "z5/multiarray/xtensor_util.hxx"


namespace nifty {
namespace tools {

    class LabelMultisetWrapper {

    public:
        typedef z5::filesystem::Dataset<uint8_t> WrappedType;
        typedef z5::Dataset WrappedBaseType;
        typedef std::shared_ptr<WrappedBaseType> PointerType;

        LabelMultisetWrapper(std::unique_ptr<WrappedBaseType> dataset) {
            ds_ = std::move(dataset);
        }


        template<class ARRAY, class COORD>
        inline void readSubarray(ARRAY & labels, const COORD & roiBegin) {

            // get the offset and shape of the request and check if it is valid
            const auto & arrShape = labels.shape();
            z5::types::ShapeType offset(roiBegin.begin(), roiBegin.end());
            z5::types::ShapeType shape(arrShape.begin(), arrShape.end());

            // get the chunks that are involved in this request
            std::vector<z5::types::ShapeType> chunkRequests;
            const auto & chunking = ds_->chunking();
            chunking.getBlocksOverlappingRoi(offset, shape, chunkRequests);

            z5::types::ShapeType offsetInRequest, requestShape, chunkShape;
            z5::types::ShapeType offsetInChunk;

            const std::size_t maxChunkSize = ds_->defaultChunkSize();
            std::size_t chunkSize = maxChunkSize;
            std::vector<uint64_t> buffer(chunkSize);

            // get the fillvalue
            const uint64_t fillValue = 0;

            // iterate over the chunks
            for(const auto & chunkId : chunkRequests) {

                bool completeOvlp = chunking.getCoordinatesInRoi(chunkId,
                                                                 offset,
                                                                 shape,
                                                                 offsetInRequest,
                                                                 requestShape,
                                                                 offsetInChunk);

                // get the view in our array
                xt::xstrided_slice_vector offsetSlice;
                z5::multiarray::sliceFromRoi(offsetSlice, offsetInRequest, requestShape);
                auto view = xt::strided_view(labels, offsetSlice);

                // check if this chunk exists, if not fill output with fill value
                if(!ds_->chunkExists(chunkId)) {
                    view = fillValue;;
                    continue;
                }

                // get the current chunk-shape
                ds_->getChunkShape(chunkId, chunkShape);
                chunkSize = std::accumulate(chunkShape.begin(), chunkShape.end(),
                                            1, std::multiplies<std::size_t>());

                // resize the buffer if necessary
                if(chunkSize != buffer.size()) {
                    buffer.resize(chunkSize);
                }

                // read the current chunk into the buffer
                readChunk(chunkId, buffer);

                // request and chunk overlap completely
                // -> we can read all the data from the chunk
                if(completeOvlp) {
                    z5::multiarray::copyBufferToView(buffer, view, labels.strides());
                }
                // request and chunk overlap only partially
                // -> we can read the chunk data only partially
                else {
                    // get a view to the part of the buffer we are interested in
                    auto fullBuffView = xt::adapt(buffer, chunkShape);
                    xt::xstrided_slice_vector bufSlice;

                    z5::multiarray::sliceFromRoi(bufSlice, offsetInChunk, requestShape);
                    auto bufView = xt::strided_view(fullBuffView, bufSlice);

                    // could also implement fast copy for this
                    // but this would be harder and might be premature optimization
                    view = bufView;
                }
            }
        }

        inline bool readChunk(const std::vector<std::size_t> & chunkId, std::vector<uint64_t> & labelVector) {

            // check if this chunk exists
            if(!ds_->chunkExists(chunkId)) {
                return false;
            }

            // get the size of this chunk and read it
            std::size_t thisSize;
            ds_->checkVarlenChunk(chunkId, thisSize);
            std::vector<uint8_t> chunkData(thisSize);
            ds_->readChunk(chunkId, &chunkData[0]);

            std::size_t chunkPos = 0;
            // find the number of labels in this chunk
            // (encoded in the first 4 bytes as signed integer '>i' in pythons struct)
            int32_t nLabels;
            std::memcpy(&nLabels, &chunkData[chunkPos], 4);
            z5::util::reverseEndiannessInplace(nLabels);
            chunkPos += 4;

            if(labelVector.size() != nLabels) {
                throw std::runtime_error("Misaligned size!");
            }

            // load the first 8 * number of labels bytes, which encode the argmax labels
            // encoded as '>q' in python's struct
            for(std::size_t labelPos = 0; labelPos < nLabels; ++labelPos) {
                uint64_t & label = labelVector[labelPos];
                std::memcpy(&label, &chunkData[chunkPos], 8);
                z5::util::reverseEndiannessInplace(label);
                chunkPos += 8;
            }
            return true;
        }

    private:
        PointerType ds_;
    };


}
}
