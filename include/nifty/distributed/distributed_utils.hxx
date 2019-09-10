#pragma once

#include "z5/util/for_each.hxx"
#include "z5/util/util.hxx"

#include "nifty/tools/blocking.hxx"
#include "nifty/distributed/graph_extraction.hxx"

#ifdef WITH_BOOST_FS
    namespace fs = boost::filesystem;
#else
    #if __GCC__ > 7
        namespace fs = std::filesystem;
    #else
        namespace fs = std::experimental::filesystem;
    #endif
#endif

namespace nifty {
namespace distributed {

    //
    // compute and serialize label overlaps
    //

    template<class LABELS, class VALUES, class OVERLAPS>
    inline void computeLabelOverlaps(const LABELS & labels,
                                     const VALUES & values,
                                     OVERLAPS & overlaps,
                                     const bool withIgnoreLabel=false,
                                     const uint64_t ignoreLabel=0) {
        CoordType shape;
        std::copy(labels.shape().begin(), labels.shape().end(), shape.begin());

        nifty::tools::forEachCoordinate(shape, [&](const CoordType & coord){
            const auto node = xtensor::read(labels, coord);
            const auto l = xtensor::read(values, coord);
            if(withIgnoreLabel && l == ignoreLabel) {
                return;
            }
            auto ovlpIt = overlaps.find(node);
            if(ovlpIt == overlaps.end()) {
                overlaps.emplace(node, std::unordered_map<uint64_t, std::size_t>{{l, 1}});
            }
            else {
                auto & ovlpMap = ovlpIt->second;
                auto mapIt = ovlpMap.find(l);
                if(mapIt == ovlpMap.end()) {
                    ovlpMap.emplace(l, 1);
                } else {
                    ++mapIt->second;
                }
            }
        });
    }


    template<class OVLPS, class DS>
    inline void serializeLabelOverlaps(const OVLPS & overlaps, DS ds,
                                       const std::vector<std::size_t> & chunkId) {
        // first determine the serialization size
        std::size_t serSize = 0;
        for(const auto & elem: overlaps) {
            // per label, we serialize labelId, number of values,
            // the values and value-counts
            serSize += 2 + 2 * elem.second.size();
        }

        // make serialize
        std::vector<uint64_t> serialization(serSize);
        std::size_t serPos = 0;
        for(const auto & elem: overlaps) {
            const uint64_t labelId = static_cast<uint64_t>(elem.first);
            serialization[serPos] = labelId;
            ++serPos;

            const uint64_t count = elem.second.size();
            serialization[serPos] = count;
            ++serPos;

            for(const auto & ovlp: elem.second) {
                const uint64_t value = static_cast<uint64_t>(ovlp.first);
                serialization[serPos] = value;
                ++serPos;

                const uint64_t count = static_cast<uint64_t>(ovlp.second);
                serialization[serPos] = count;
                ++serPos;
            }
        }

        // write serialization
        ds->writeChunk(chunkId, &serialization[0], true, serSize);
    }


    template<class LABELS, class VALUES>
    inline void computeAndSerializeLabelOverlaps(const LABELS & labels,
                                                 const VALUES & values,
                                                 const std::string & path,
                                                 const std::string & key,
                                                 const std::vector<std::size_t> & chunkId,
                                                 const bool withIgnoreLabel=false,
                                                 const uint64_t ignoreLabel=0) {
        typedef typename LABELS::value_type LabelType;
        typedef typename VALUES::value_type ValueType;
        typedef std::unordered_map<ValueType, std::size_t> OverlapType;
        // extract the overlaps
        std::unordered_map<LabelType, OverlapType> overlaps;
        computeLabelOverlaps(labels, values, overlaps, withIgnoreLabel, ignoreLabel);

        // serialize the overlaps
        if(overlaps.size() > 0) {
            const z5::filesystem::handle::File file(path);
            auto ds = z5::openDataset(file, key);
            serializeLabelOverlaps(overlaps, std::move(ds), chunkId);
        }
    }


    template<class OVLP>
    inline void deserializeOverlapsFromData(const std::vector<uint64_t> & chunkOverlaps,
                                            uint64_t & maxLabelId,
                                            OVLP & out,
                                            const uint64_t labelBegin=0,
                                            const uint64_t labelEnd=0) {
        typedef typename OVLP::value_type::second_type OverlapType;

        const bool checkNodeRange = labelBegin != labelEnd;

        const std::size_t chunkSize = chunkOverlaps.size();
        std::size_t pos = 0;
        while(pos < chunkSize) {
            const uint64_t labelId = chunkOverlaps[pos];
            ++pos;

            const bool inRange = checkNodeRange ? (labelId >= labelBegin && labelId < labelEnd) : true;

            if(inRange) {
                // std::cout << "label " <<  labelId << " is in range"  << std::endl;
                if(labelId > maxLabelId) {
                    maxLabelId = labelId;
                }

                const uint64_t nValues = chunkOverlaps[pos];
                ++pos;

                auto labelIt = out.find(labelId);
                if(labelIt == out.end()) {
                    labelIt = out.emplace(std::make_pair(labelId, OverlapType())).first;
                }

                auto & ovlps = labelIt->second;
                for(std::size_t i = 0; i < nValues; ++i) {
                    const uint64_t value = chunkOverlaps[pos];
                    ++pos;

                    auto valIt = ovlps.find(value);
                    if(valIt == ovlps.end()) {
                        valIt = ovlps.emplace(std::make_pair(value, 0)).first;
                    }

                    const uint64_t count = chunkOverlaps[pos];
                    ++pos;

                    valIt->second += count;
                }
            } else {
                // std::cout << "label " <<  labelId << " is out of range" << std::endl;
                const uint64_t nValues = chunkOverlaps[pos];
                ++pos;
                for(std::size_t i = 0; i < nValues; ++i) {
                    const uint64_t value = chunkOverlaps[pos];
                    ++pos;
                    const uint64_t count = chunkOverlaps[pos];
                    ++pos;
                }
            }
        }
    }


    template<class OVLP>
    inline uint64_t deserializeOverlapChunk(const std::string & path,
                                            const std::string & key,
                                            const std::vector<std::size_t> & chunkId,
                                            OVLP & out) {
        const z5::filesystem::handle::File file(path);
        auto ds = z5::openDataset(file, key);
        uint64_t maxLabelId = 0;

        // read this chunk's data (if present)
        if(ds->chunkExists(chunkId)) {
            std::size_t chunkSize;
            ds->checkVarlenChunk(chunkId, chunkSize);
            std::vector<uint64_t> chunkOverlaps(chunkSize);
            ds->readChunk(chunkId, &chunkOverlaps[0]);

            // deserialize the data
            deserializeOverlapsFromData(chunkOverlaps, maxLabelId, out);
        }
        return maxLabelId;
    }


    inline void mergeAndSerializeOverlaps(const std::string & inputPath,
                                          const std::string & inputKey,
                                          const std::string & outputPath,
                                          const std::string & outputKey,
                                          const bool max_overlap,
                                          const int numberOfThreads,
                                          const uint64_t labelBegin,
                                          const uint64_t labelEnd,
                                          const uint64_t ignoreLabel=0,
                                          const bool serializeCount=false) {

        // std::cout << "merge and serialize from " << labelBegin << " to " << labelEnd << std::endl;
        typedef std::unordered_map<uint64_t, std::size_t> OverlapType;
        typedef std::unordered_map<uint64_t, OverlapType> LabelToOverlaps;

        std::vector<LabelToOverlaps> threadData(numberOfThreads);
        std::vector<uint64_t> threadMax(numberOfThreads, 0);

        const z5::filesystem::handle::File inFile(inputPath);
        const z5::filesystem::handle::File outFile(outputPath);
        auto inputDs = z5::openDataset(inFile, inputKey);
        z5::util::parallel_for_each_chunk(*inputDs,
                                          numberOfThreads,
                                          [&threadData,
                                           &threadMax,
                                           labelBegin,
                                           labelEnd](const int tId,
                                                     const z5::Dataset & ds,
                                                     const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            if(!ds.chunkExists(chunkCoord)) {
                return;
            }
            std::size_t chunkSize;
            ds.checkVarlenChunk(chunkCoord, chunkSize);
            std::vector<uint64_t> chunkOverlaps(chunkSize);
            ds.readChunk(chunkCoord, &chunkOverlaps[0]);

            // deserialize the data
            auto & thisData = threadData[tId];
            uint64_t & thisMax = threadMax[tId];
            deserializeOverlapsFromData(chunkOverlaps, thisMax, thisData,
                                        labelBegin, labelEnd);

        });
        // std::cout << "Load from chunks done" << std::endl;

        // find the upper label bound
        // const uint64_t nLabels = *std::max_element(threadMax.begin(), threadMax.end()) + 1;
        const uint64_t nLabels = labelEnd - labelBegin;

        // merge the thread data
        std::vector<OverlapType> overlaps(nLabels);
        // std::cout << "Merge" << std::endl;
        nifty::parallel::parallel_foreach(numberOfThreads,
                                          nLabels,
                                          [&threadData,
                                           &overlaps,
                                           numberOfThreads,
                                           labelBegin](const int t,
                                                       const uint64_t labelId){
            auto & ovlp = overlaps[labelId];
            for(int tId = 0; tId < numberOfThreads; ++tId) {
                const auto & src = threadData[tId];
                const auto & srcIt = src.find(labelId + labelBegin);
                if(srcIt == src.end()) {
                    continue;
                }
                const auto & srcOvlps = srcIt->second;
                for(const auto & srcElem: srcOvlps) {
                    const uint64_t value = srcElem.first;
                    const uint64_t count = srcElem.second;

                    auto outIt = ovlp.find(value);
                    if(outIt == ovlp.end()) {
                        outIt = ovlp.emplace(std::make_pair(value, 0)).first;
                    }
                    outIt->second += count;
                }
            }
        });
        // std::cout << "Merge done" << std::endl;

        // serialzie the result
        if(max_overlap && serializeCount) {  // serialize the label with maximum overlap and the associated count
            // find the maximum overlap value for each label
            xt::xtensor<uint64_t, 2> out = xt::zeros<uint64_t>({nLabels, static_cast<uint64_t>(2)});
            nifty::parallel::parallel_foreach(numberOfThreads,
                                              nLabels,
                                              [&out,
                                               &overlaps,
                                               ignoreLabel](const int t,
                                                            const uint64_t labelId){
                const auto & ovlp = overlaps[labelId];
                // NOTE we initialise by the ignore label here, because if we have an ignore-label
                // and a node ONLY overalps with ignore label, its overlap vector
                // will be empty and we need to indicate this
                uint64_t maxOvlpValue = ignoreLabel;
                std::size_t maxOvlp = 0;
                for(const auto & elem: ovlp) {
                    if(elem.second > maxOvlp) {
                        maxOvlp = elem.second;
                        maxOvlpValue = elem.first;
                    }
                }
                out(labelId, 0) = maxOvlpValue;
                out(labelId, 1) = maxOvlp;
            });

            auto dsOut = z5::openDataset(outFile, outputKey);
            const std::vector<std::size_t> zero2Coord({labelBegin, 0});
            z5::multiarray::writeSubarray<uint64_t>(dsOut, out,
                                                    zero2Coord.begin(), numberOfThreads);
        } else if(max_overlap) { // serialize the label with maximum overlap
            // find the maximum overlap value for each label
            xt::xtensor<uint64_t, 1> out = xt::zeros<uint64_t>({nLabels});
            nifty::parallel::parallel_foreach(numberOfThreads,
                                              nLabels,
                                              [&out,
                                               &overlaps,
                                               ignoreLabel](const int t,
                                                            const uint64_t labelId){
                const auto & ovlp = overlaps[labelId];
                // NOTE we initialise by the ignore label here, because if we have an ignore-label
                // and a node ONLY overalps with ignore label, its overlap vector
                // will be empty and we need to indicate this
                uint64_t maxOvlpValue = ignoreLabel;
                std::size_t maxOvlp = 0;
                for(const auto & elem: ovlp) {
                    if(elem.second > maxOvlp) {
                        maxOvlp = elem.second;
                        maxOvlpValue = elem.first;
                    }
                }
                out(labelId) = maxOvlpValue;
            });

            auto dsOut = z5::openDataset(outFile, outputKey);
            const std::vector<std::size_t> zero1Coord({labelBegin});
            z5::multiarray::writeSubarray<uint64_t>(dsOut, out,
                                                    zero1Coord.begin(), numberOfThreads);

        } else {  // serialize the merged overlap dict to a single chunk
            // merge the dicts (better not parallelize)
            auto & out = threadData[0];
            for(int tId = 1; tId < numberOfThreads; ++tId) {
                const auto & src = threadData[tId];
                for(const auto & elem: src) {
                    const uint64_t labelId = elem.first;
                    const auto & srcOvlp = elem.second;

                    auto outIt = out.find(labelId);
                    if(outIt == out.end()) {
                        outIt = out.emplace(std::make_pair(labelId, OverlapType())).first;
                    }

                    auto & outOvlp = outIt->second;
                    for(const auto & ovlp: srcOvlp) {
                        const uint64_t value = ovlp.first;
                        const uint64_t count = ovlp.second;

                        auto ovlpIt = outOvlp.find(value);
                        if(ovlpIt == outOvlp.end()) {
                            ovlpIt = outOvlp.emplace(std::make_pair(value, 0)).first;
                        }
                        ovlpIt->second += count;
                    }
                }
            }

            // get the correct chunk id
            auto ds = z5::openDataset(outFile, outputKey);
            const std::size_t chunkSize = ds->defaultChunkShape(0);
            const std::vector<std::size_t> chunkId = {labelBegin / chunkSize};
            serializeLabelOverlaps(out, std::move(ds), chunkId);
        }
    }


    //
    // compute and serialize label-to-block-mapping
    //

    template<class OUT, class THREAD_DATA>
    inline void mergeBlockMapping(const THREAD_DATA & perThreadData,
                                  OUT & mapping, const int numberOfThreads) {
        // merge the label data into output vector
        const std::size_t numberOfLabels = mapping.size();
        nifty::parallel::parallel_foreach(numberOfThreads, numberOfLabels,
                                          [&](const int t,
                                              const uint64_t labelId){
            auto & out = mapping[labelId];
            for(int threadId = 0; threadId < numberOfThreads; ++threadId) {
                auto & threadData = perThreadData[threadId];
                auto it = threadData.find(labelId);
                if(it != threadData.end()) {
                    const auto & copyIds = it->second;
                    out.reserve(out.size() + copyIds.size());
                    out.insert(out.end(), copyIds.begin(), copyIds.end());
                }
            }
            std::sort(out.begin(), out.end());
        });

    }


    template<class OUT>
    inline void getBlockMapping(const std::string & inputPath,
                                const std::string & inputKey,
                                const int numberOfThreads,
                                OUT & mapping) {

        const z5::filesystem::handle::File file(inputPath);
        auto inputDs = z5::openDataset(file, inputKey);

        // we store the mapping of labels to blocks extracted for each thread in an unordered map
        // of vectors
        typedef std::unordered_map<uint64_t, std::vector<std::size_t>> PerThread;
        std::vector<PerThread> perThreadData(numberOfThreads);

        const auto & blocking = inputDs->chunking();
        z5::util::parallel_for_each_chunk(*inputDs,
                                          numberOfThreads,
                                          [&perThreadData,
                                           &blocking](const int tId,
                                                      const z5::Dataset & ds,
                                                      const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            if(!ds.chunkExists(chunkCoord)) {
                return;
            }

            std::size_t chunkSize;
            ds.checkVarlenChunk(chunkCoord, chunkSize);
            std::vector<uint64_t> labelsInChunk(chunkSize);
            ds.readChunk(chunkCoord, &labelsInChunk[0]);

            // get the (1d) id of the chunk
            const std::size_t chunkId = blocking.blockCoordinatesToBlockId(chunkCoord);

            // add the chunkId to all the labels we have found in this chunk
            auto & threadData = perThreadData[tId];
            for(const uint64_t labelId: labelsInChunk) {
                auto it = threadData.find(labelId);
                if(it == threadData.end()) {
                    threadData.insert(std::make_pair(labelId,
                                                     std::vector<std::size_t>({chunkId})));
                } else {
                    it->second.push_back(chunkId);
                }
            }
        });

        mergeBlockMapping(perThreadData, mapping, numberOfThreads);
    }


    template<class OUT>
    inline void getBlockMappingWithRoi(const std::string & inputPath,
                                       const std::string & inputKey,
                                       const int numberOfThreads,
                                       OUT & mapping,
                                       const std::vector<std::size_t> & roiBegin,
                                       const std::vector<std::size_t> & roiEnd) {

        const z5::filesystem::handle::File file(inputPath);
        auto inputDs = z5::openDataset(file, inputKey);

        // we store the mapping of labels to blocks extracted for each thread in an unordered map
        // of vectors
        typedef std::unordered_map<uint64_t, std::vector<std::size_t>> PerThread;
        std::vector<PerThread> perThreadData(numberOfThreads);

        const auto & blocking = inputDs->chunking();
        z5::util::parallel_for_each_chunk_in_roi(*inputDs,
                                                 roiBegin,
                                                 roiEnd,
                                                 numberOfThreads,
                                                 [&perThreadData,
                                                  &blocking](const int tId,
                                                             const z5::Dataset & ds,
                                                             const z5::types::ShapeType & chunkCoord){
            // read this chunk's data (if present)
            if(!ds.chunkExists(chunkCoord)) {
                return;
            }

            std::size_t chunkSize;
            ds.checkVarlenChunk(chunkCoord, chunkSize);
            std::vector<uint64_t> labelsInChunk(chunkSize);
            ds.readChunk(chunkCoord, &labelsInChunk[0]);

            // get the (1d) id of the chunk
            const std::size_t chunkId = blocking.blockCoordinatesToBlockId(chunkCoord);

            // add the chunkId to all the labels we have found in this chunk
            auto & threadData = perThreadData[tId];
            for(const uint64_t labelId: labelsInChunk) {
                auto it = threadData.find(labelId);
                if(it == threadData.end()) {
                    threadData.insert(std::make_pair(labelId,
                                                     std::vector<std::size_t>({chunkId})));
                } else {
                    it->second.push_back(chunkId);
                }
            }
        });

        mergeBlockMapping(perThreadData, mapping, numberOfThreads);
    }


    template<class OUT>
    inline void serializeMappingChunks(const std::string & inputPath,
                                       const std::string & inputKey,
                                       const std::string & outputPath,
                                       const std::string & outputKey,
                                       const OUT & mapping,
                                       const int numberOfThreads) {
        // open the input and output datasets
        const z5::filesystem::handle::File inFile(inputPath);
        const z5::filesystem::handle::File outFile(outputPath);
        auto inputDs = z5::openDataset(inFile, inputKey);
        auto outputDs = z5::openDataset(outFile, outputKey);
        const auto & blocking = inputDs->chunking();

        const std::size_t numberOfLabels = mapping.size();
        const std::vector<std::size_t> idRoiBegin = {0};
        const std::vector<std::size_t> idRoiEnd = {numberOfLabels};
        z5::util::parallel_for_each_chunk_in_roi(*outputDs,
                                                 idRoiBegin,
                                                 idRoiEnd,
                                                 numberOfThreads,
                                                 [&mapping,
                                                  &blocking,
                                                  numberOfLabels](const int t,
                                                                  const z5::Dataset & ds,
                                                                  const z5::types::ShapeType & idChunk){

            const auto & idBlocking = ds.chunking();
            // get the begin and end in id-space for this chunk
            std::vector<std::size_t> idBegin, idEnd;
            idBlocking.getBlockBeginAndEnd(idChunk, idBegin, idEnd);
            idEnd[0] = std::min(idEnd[0], numberOfLabels);

            // calculate the serialization size for this chunk in byte
            std::size_t serSize = 0;
            for(int64_t labelId = idBegin[0]; labelId < idEnd[0]; ++labelId) {
                const std::size_t nBlocks = mapping[labelId].size();
                // for every label that is present in at least one block, we serialize:
                // labelId as int64 = 8 byte
                // number of blocks as int32 = 4 byte
                // 6 int64 coordinates for each block = 6 * 8 * nBlocks = 48 * nBlocks
                if(nBlocks > 0) {
                    serSize += 12 + nBlocks * 48;
                }
            }

            if(serSize == 0) {
                return;
            }

            std::vector<std::size_t> chunkBegin, chunkEnd;
            // make serialzation
            char * byteSerialization = new char[serSize];
            char * serPointer = byteSerialization;
            for(int64_t labelId = idBegin[0]; labelId < idEnd[0]; ++labelId) {
                const auto & blockList = mapping[labelId];
                int32_t nBlocks = static_cast<int32_t>(blockList.size());
                if(nBlocks > 0) {
                    // copy labelId, numberOfBlocks into the serialization buffer
                    int64_t labelIdOut = labelId;

                    // for some reason, this is not in the n5 default endianness,
                    // so we need to reverse the endidianness for everything here
                    z5::util::reverseEndiannessInplace(labelIdOut);
                    memcpy(serPointer, &labelIdOut, 8);
                    serPointer += 8;

                    z5::util::reverseEndiannessInplace(nBlocks);
                    memcpy(serPointer, &nBlocks, 4);
                    serPointer += 4;

                    // serialize the coordinates for all blocks in blocklist
                    for(const std::size_t chunkId: blockList) {
                        blocking.getBlockBeginAndEnd(chunkId, chunkBegin, chunkEnd);

                        // NOTE, java has axis order XYZ, we have ZYX that's why we revert
                        // also, we report the end coordinates (= max + 1), java expects max
                        std::array<int64_t, 6> blockSer = {static_cast<int64_t>(chunkBegin[2]),
                                                           static_cast<int64_t>(chunkBegin[1]),
                                                           static_cast<int64_t>(chunkBegin[0]),
                                                           static_cast<int64_t>(chunkEnd[2] - 1),
                                                           static_cast<int64_t>(chunkEnd[1] - 1),
                                                           static_cast<int64_t>(chunkEnd[0] - 1)};

                        for(int64_t bc : blockSer) {
                            z5::util::reverseEndiannessInplace(bc);
                            memcpy(serPointer, &bc, 8);
                            serPointer += 8;
                        }
                    }
                }
            }
            // write serialization to this chunk
            ds.writeChunk(idChunk, byteSerialization, true, serSize);
            delete[] byteSerialization;
        });

    }


    inline void serializeBlockMapping(const std::string & inputPath,
                                      const std::string & inputKey,
                                      const std::string & outputPath,
                                      const std::string & outputKey,
                                      const std::size_t numberOfLabels,
                                      const int numberOfThreads,
                                      const std::vector<std::size_t> & roiBegin=std::vector<std::size_t>(),
                                      const std::vector<std::size_t> & roiEnd=std::vector<std::size_t>()) {

        // iterate over the input in parallel and map block-ids to label ids
        std::vector<std::vector<std::size_t>> mapping(numberOfLabels);
        if(roiBegin.size() == 0) {
            getBlockMapping(inputPath, inputKey, numberOfThreads, mapping);
        } else if(roiBegin.size() == 3) {
            getBlockMappingWithRoi(inputPath, inputKey, numberOfThreads, mapping, roiBegin, roiEnd);
        } else {
            throw std::runtime_error("Invalid ROI");
        }
        serializeMappingChunks(inputPath, inputKey, outputPath, outputKey,
                               mapping, numberOfThreads);
    }


    // take block mapping serialization and return the map
    inline void formatBlockMapping(const std::vector<char> & input,
                                   std::map<std::uint64_t, std::vector<std::array<int64_t, 6>>> & mapping) {
        const std::size_t byteSize = input.size();
        const char * serPointer = &input[0];

        int64_t labelId;
        int32_t nBlocks;
        std::array<int64_t, 6> coords;
        while(std::distance(&input[0], serPointer) < byteSize) {
            memcpy(&labelId, serPointer, 8);
            z5::util::reverseEndiannessInplace(labelId);
            serPointer += 8;

            memcpy(&nBlocks, serPointer, 4);
            z5::util::reverseEndiannessInplace(nBlocks);
            serPointer += 4;

            std::vector<std::array<int64_t, 6>> coordList;
            for(int i = 0; i < nBlocks; ++i) {
                for(int c = 0; c < 6; ++c) {
                    memcpy(&coords[c], serPointer, 8);
                    serPointer += 8;
                }
                z5::util::reverseEndiannessInplace<int64_t>(coords.begin(), coords.end());
                coordList.push_back(coords);
            }
            mapping[static_cast<uint64_t>(labelId)] = coordList;
        }
    }


    inline void readBlockMapping(const std::string & path, const std::string & key,
                                 const std::vector<std::size_t> chunkId,
                                 std::map<std::uint64_t, std::vector<std::array<int64_t, 6>>> & mapping) {
        const z5::filesystem::handle::File file(path);
        auto ds = z5::openDataset(file, key);
        if(ds->chunkExists(chunkId)) {
            std::size_t chunkSize;
            ds->checkVarlenChunk(chunkId, chunkSize);
            std::vector<char> out(chunkSize);
            ds->readChunk(chunkId, &out[0]);
            formatBlockMapping(out, mapping);
        }
    }

}
}
