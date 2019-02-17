#pragma once

#include "z5/util/for_each.hxx"
#include "z5/util/util.hxx"

#include "nifty/tools/blocking.hxx"
#include "nifty/distributed/graph_extraction.hxx"

namespace fs = boost::filesystem;

namespace nifty {
namespace distributed {

    //
    // compute and serialize label overlaps
    //

    template<class LABELS, class MORPHOLOGY>
    inline void computeMorphology(const LABELS & labels,
                                  const std::vector<std::size_t> coordinateOffset,
                                  MORPHOLOGY & morphology) {
        CoordType shape;
        std::copy(labels.shape().begin(), labels.shape().end(), shape.begin());

        nifty::tools::forEachCoordinate(shape, [&](const CoordType & coord){
            const auto labelId = xtensor::read(labels, coord);
            auto it = morphology.find(labelId);
            if(it == morphology.end()) {
                // if this label-id is not yet in the morphology, emplace a new element
                // with size initialized to 1 and com, min- and max-coord initialized to current coord
                morphology.emplace(std::make_pair(labelId,
                                                  std::array<double, 10>({1.,
                                                                          static_cast<double>(coord[0]), static_cast<double>(coord[1]), static_cast<double>(coord[2]),
                                                                          static_cast<double>(coord[0]), static_cast<double>(coord[1]), static_cast<double>(coord[2]),
                                                                          static_cast<double>(coord[0]), static_cast<double>(coord[1]), static_cast<double>(coord[2])})));
            } else {
                // otherwise merge with current morphology
                auto & morph = it->second;
                // center of mass: average weighted by pixel size
                // min coord: minimum of coord and new coord
                // max coord: maximum of coord and new coord
                const double prevSize = morph[0];
                for(int d = 0; d < 3; ++d) {
                    morph[d + 1] = (prevSize * morph[d + 1] + coord[d]) / (prevSize + 1);
                    morph[d + 4] = std::min(morph[d + 4], static_cast<double>(coord[d]));
                    morph[d + 7] = std::max(morph[d + 7], static_cast<double>(coord[d]));
                }
                // increase size by 1
                ++morph[0];
            }
        });

        // add the offset coordinate
        for(auto & morphElem: morphology) {
            auto & morph = morphElem.second;
            for(int d = 0; d < 3; ++d) {
                morph[d + 1] += coordinateOffset[d];
                morph[d + 4] += coordinateOffset[d];
                morph[d + 7] += coordinateOffset[d];
            }
        }
    }


    template<class MORPHOLOGY>
    inline void serializeMorphology(const MORPHOLOGY & morphology,
                                    const std::string & outPath,
                                    const std::vector<std::size_t> & chunkId) {
        // first determine the serialization size = 11 elems per label
        const std::size_t serSize = morphology.size() * 11;

        // make serialize
        std::vector<double> serialization(serSize);
        auto serIt = serialization.begin();
        for(const auto & elem: morphology) {
            *serIt = static_cast<double>(elem.first);
            ++serIt;

            auto & morph = elem.second;
            std::copy(morph.begin(), morph.end(), serIt);
            std::advance(serIt, morph.size());
        }

        // write serialization
        auto ds = z5::openDataset(outPath);
        ds->writeChunk(chunkId, &serialization[0], true, serSize);
    }


    template<class LABELS>
    inline void computeAndSerializeMorphology(const LABELS & labels,
                                              const std::vector<std::size_t> & coordinateOffset,
                                              const std::string & outPath,
                                              const std::vector<std::size_t> & chunkId) {
        typedef typename LABELS::value_type LabelType;
        // we keep track of 10 values in the morphology:
        // 1.) size in pixels
        // 2.) - 4.) center of mass
        // 5.) - 7.) min coordinate
        // 8.) - 10.) max coordinate
        typedef std::array<double, 10> MorphologyType;
        std::unordered_map<uint64_t, MorphologyType> morphology;
        // extract the morphology
        computeMorphology(labels, coordinateOffset, morphology);

        // serialize the morphology
        if(morphology.size() > 0) {
            serializeMorphology(morphology, outPath, chunkId);
        }
    }


    template<class MORPHOLOGY>
    inline void mergeMorphology(const std::vector<double> & morphologyIn,
                                MORPHOLOGY & out,
                                const uint64_t labelBegin=0,
                                const uint64_t labelEnd=0) {
        const bool checkNodeRange = labelBegin != labelEnd;
        const std::size_t chunkSize = morphologyIn.size();
        std::size_t pos = 0;
        while(pos < chunkSize) {
            const double labelId = morphologyIn[pos];
            ++pos;

            const bool inRange = checkNodeRange ? (labelId >= labelBegin && labelId < labelEnd) : true;

            if(inRange) {
                const std::size_t labelIndex = labelId - labelBegin;
                const double prevSize = out(labelIndex, 1);
                const double thisSize = morphologyIn[pos];
                // add up sizes size
                out(labelIndex, 1) += thisSize;
                ++pos;
                // merge the com coordinate
                for(int d = 0; d < 3; ++d) {
                    out(labelIndex, 2 + d) = (prevSize * out(labelIndex, 2 + d) + thisSize * morphologyIn[pos]) / (prevSize + thisSize);
                    ++pos;
                }
                // merge the min coordinate
                for(int d = 0; d < 3; ++d) {
                    out(labelIndex, 5 + d) = std::min(out(labelIndex, 5 + d), morphologyIn[pos]);
                    ++pos;
                }
                // merge the max coordinate
                for(int d = 0; d < 3; ++d) {
                    out(labelIndex, 8 + d) = std::max(out(labelIndex, 8 + d), morphologyIn[pos]);
                    ++pos;
                }
            } else {
                pos += 10;
            }
        }
    }

    inline void mergeAndSerializeMorphology(const std::string & inputPath,
                                            const std::string & outputPath,
                                            const uint64_t labelBegin,
                                            const uint64_t labelEnd) {

        // declare and initialize output data
        typedef typename xt::xtensor<double, 2>::shape_type ShapeType;
        const std::size_t nLabels = labelEnd - labelBegin;
        ShapeType outShape = {nLabels, 11};
        xt::xtensor<double, 2> out(outShape);

        //
        for(std::size_t labelId = 0; labelId < nLabels; ++labelId) {
            // set label id to label id
            out(labelId, 0) = labelId + labelBegin;
            // initialize size to 0
            out(labelId, 1) = 0;
            // initialize com to 0, min to double max val, max to 0
            for(int d = 0; d < 3; ++d) {
                out(labelId, 2 + d) = 0;
                out(labelId, 5 + d) = std::numeric_limits<double>::max();
                out(labelId, 8 + d) = 0;
            }
        }


        auto inputDs = z5::openDataset(inputPath);
        z5::util::parallel_for_each_chunk(*inputDs, 1,
                                          [&out,
                                           labelBegin,
                                           labelEnd](const int tId,
                                                     const z5::Dataset & ds,
                                                     const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            z5::handle::Chunk chunk(ds.handle(), chunkCoord, ds.isZarr());
            if(!chunk.exists()) {
                return;
            }
            bool isVarlen;
            const std::size_t chunkSize = ds.getDiscChunkSize(chunkCoord, isVarlen);
            std::vector<double> morphologyIn(chunkSize);
            ds.readChunk(chunkCoord, &morphologyIn[0]);
            mergeMorphology(morphologyIn, out,
                            labelBegin, labelEnd);

        });

        auto dsOut = z5::openDataset(outputPath);
        const std::vector<std::size_t> offset({labelBegin, 0});
        z5::multiarray::writeSubarray<double>(dsOut, out, offset.begin());
    }

}
}
