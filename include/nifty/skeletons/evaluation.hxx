#include "nifty/z5/z5.hxx"
#include "nifty/parallel/threadpool.hxx"


namespace fs = boost::filesystem;
namespace nifty {
namespace skeletons {

    //
    // TODO maybe refactor this as a class ?

    // store coordinates of a single (sub-) skeleton
    typedef xt::xtensor<size_t, 2> CoordinateArray;
    typedef typename CoordinateArray::shape_type ArrayShape;
    // store tmp coordinates as vector
    typedef std::vector<size_t> CoordinateVector;
    // store (sub-) skeleton id to coordinates assignment
    typedef std::map<size_t, CoordinateArray> SkeletonStorage;
    // store (sub-) skeletons for all blocks
    typedef std::map<size_t, SkeletonStorage> SkeletonBlockStorage;
    // assignmet of skeleton nodes to segmentation nodes
    typedef std::unordered_map<size_t, size_t> SkeletonNodeAssignment;
    typedef std::map<size_t, SkeletonNodeAssignment> SkeletonDictionary;


    // TODO de-spaghettify
    inline void groupSkeletonBlocks(const std::string & segmentationPath,
                                    const std::string & skeletonTopFolder,
                                    const std::vector<size_t> & skeletonIds,
                                    parallel::ThreadPool & threadpool,
                                    SkeletonBlockStorage & out,
                                    std::vector<size_t> & nonEmptyChunks) {

        std::vector<size_t> zeroCoord = {0, 0};

        // open the segmentation dataset and get the shape and chunks
        // std::cout << "open segmentation" << std::endl;
        auto segmentation = z5::openDataset(segmentationPath);
        // std::cout << "done" << std::endl;
        const size_t nChunks = segmentation->numberOfChunks();

        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // go over all skeletons in parallel
        // and extract the sub skeletons for each block
        const size_t nSkeletons = skeletonIds.size();
        const size_t nThreads = threadpool.nThreads();

        // temporary block data for the threads
        std::vector<SkeletonBlockStorage> perThreadData(nThreads);

        // go over all skeletons in parallel and extract the parts overlapping with chunks
        parallel::parallel_foreach(threadpool, nSkeletons, [&](const int tId, const size_t skeletonIndex){
            // open the coordinate dataset for this particular skeleton
            const size_t skeletonId = skeletonIds[skeletonIndex];
            fs::path skeletonPath(skeletonTopFolder);
            skeletonPath /= std::to_string(skeletonId);
            skeletonPath /= "coordinates";
            auto coordinateSet = z5::openDataset(skeletonPath.string());
            const size_t nPoints = coordinateSet->shape(0);
            // load the coordinate data
            ArrayShape coordShape = {nPoints, coordinateSet->shape(1)};
            CoordinateArray coords(coordShape);
            z5::multiarray::readSubarray<uint64_t>(coordinateSet, coords, zeroCoord.begin());
            // loop over all coordinates, and find the corresponding chunk
            std::map<size_t, std::vector<CoordinateVector>> blocksToCoordinates; // maping of coordinates to chunks
            for(size_t point = 0; point < nPoints; ++point) {
                // we assume to have four entries per coordinate, the first is the skeleton node index,
                // the rest are the coordinates

                CoordinateVector coordinate = {coords(point, 1), coords(point, 2), coords(point, 3)};
                // get the indices of this chunk and concert them to a flat index
                std::vector<size_t> chunkIds;
                segmentation->coordinateToChunkId(coordinate, chunkIds);
                size_t chunkId = 0;
                for(unsigned dim = 0; dim < 3; ++dim) {
                    chunkId += chunkIds[dim] * chunkStrides[dim];
                }

                // prepend the skeleton index to the coordinate
                coordinate.insert(coordinate.begin(), coords(point, 0));
                // see if we already have coordinates in this block and add them
                auto blockIt = blocksToCoordinates.find(chunkId);
                if(blockIt == blocksToCoordinates.end()) {
                    std::vector<CoordinateVector> newCoordinates = {coordinate};
                    blocksToCoordinates.insert(blockIt, std::make_pair(chunkId, newCoordinates));
                } else {
                    blockIt->second.emplace_back(coordinate);
                }
            }

            // write the result to our per thread data
            auto & threadData = perThreadData[tId];
            for(const auto & elem : blocksToCoordinates) {
                const size_t chunkId = elem.first;

                // go from flat chunk index to chunk indices
                std::vector<size_t> chunkIds(3);
                size_t tmpIdx = chunkId;
                for(unsigned dim = 0; dim < 3; ++dim) {
                    chunkIds[dim] = tmpIdx / chunkStrides[dim];
                    tmpIdx -= chunkIds[dim] * chunkStrides[dim];
                }

                // find the offset of this chunk
                std::vector<size_t> chunkOffset;
                segmentation->getChunkOffset(chunkIds, chunkOffset);

                auto threadIt = threadData.find(chunkId);
                if(threadIt == threadData.end()) {
                    threadIt = threadData.insert(threadIt,
                                                 std::make_pair(chunkId, SkeletonStorage()));
                }

                // crop and copy coordinates
                ArrayShape coordShape = {elem.second.size(), 4};
                CoordinateArray coordArray(coordShape);
                size_t ii = 0;
                for(const auto & coord : elem.second) {
                    coordArray(ii, 0) = coord[0];
                    coordArray(ii, 1) = coord[1] - chunkOffset[0];
                    coordArray(ii, 2) = coord[2] - chunkOffset[1];
                    coordArray(ii, 3) = coord[3] - chunkOffset[2];
                    ++ii;
                }
                // insert in thread data
                threadIt->second.insert(std::make_pair(skeletonId, coordArray));
            }

        });

        // find all nonempty chunks
        std::set<size_t> nonEmptyChunkSet;
        for(size_t t = 0; t < nThreads; ++t) {
            auto & threadData = perThreadData[t];
            for(auto it = threadData.begin(); it != threadData.end(); ++it) {
                nonEmptyChunkSet.insert(it->first);
            }
        }
        const size_t nNonEmpty = nonEmptyChunkSet.size();
        nonEmptyChunks.resize(nNonEmpty);
        std::copy(nonEmptyChunkSet.begin(), nonEmptyChunkSet.end(), nonEmptyChunks.begin());

        // create empty entries for all non-empty chunks
        for(const size_t chunkId : nonEmptyChunks) {
            out.insert(std::make_pair(chunkId, SkeletonStorage()));
        }

        // go over all non-empty blocks in parallel and assemble all the sub-skeletons
        parallel::parallel_foreach(threadpool, nNonEmpty, [&](const int tId, const size_t nonEmptyIndex){
            const size_t chunkId = nonEmptyChunks[nonEmptyIndex];
            auto & outChunk = out[chunkId];

            // go over all the thread data and insert sub-skeletons of this chunk
            for(size_t t = 0; t < nThreads; ++t) {
                auto & threadData = perThreadData[t];
                auto threadIt = threadData.find(chunkId);
                if(threadIt == threadData.end()) {
                    continue;
                }
                outChunk.insert(threadIt->second.begin(), threadIt->second.end());
            }
        });

    }


    inline void extractNodeAssignmentsForBlock(const std::string & segmentationPath,
                                               const size_t chunkId,
                                               const SkeletonStorage & skeletons,
                                               SkeletonDictionary & out) {
        // std::vector<size_t> zeroCoord = {0, 0, 0};

        // open the segmentation dataset and get the shape and chunks
        auto segmentation = z5::openDataset(segmentationPath);
        const size_t nChunks = segmentation->numberOfChunks();

        // we could do all that outside of the function only once,
        // but it shouldn't matter at all for performance

        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // go from the chunk id to chunk index vector
        std::vector<size_t> chunkIds(3);
        size_t tmpIdx = chunkId;
        for(unsigned dim = 0; dim < 3; ++dim) {
            chunkIds[dim] = tmpIdx / chunkStrides[dim];
            tmpIdx -= chunkIds[dim] * chunkStrides[dim];
        }

        // load the chunk data
        std::vector<size_t> chunkOffset;
        segmentation->getChunkOffset(chunkIds, chunkOffset);
        std::vector<size_t> chunkShape;
        segmentation->getChunkShape(chunkIds, chunkShape);

        typedef typename xt::xtensor<uint64_t, 3> ::shape_type LabelsShape;
        LabelsShape labelsShape = {chunkShape[0], chunkShape[1], chunkShape[2]};
        xt::xtensor<uint64_t, 3> labels(labelsShape);
        z5::multiarray::readSubarray<uint64_t>(segmentation, labels, chunkOffset.begin());

        // iterate over all the skeletons in this chunks and extract the node assignment
        for(const auto & skel : skeletons) {
            const size_t skeletonId = skel.first;

            // check if the skeleton dictionary (out) already has this skeleton id
            auto skelIt = out.find(skeletonId);
            if(skelIt == out.end()) {
                skelIt = out.insert(skelIt, std::make_pair(skeletonId, SkeletonNodeAssignment()));
            }
            auto & nodeAssignment = skelIt->second;

            const auto & skelCoordinates = skel.second;
            const size_t nPoints = skelCoordinates.shape()[0];
            for(size_t point = 0; point < nPoints; ++point) {
                nodeAssignment[skelCoordinates(point, 0)] = labels(skelCoordinates(point, 1),
                                                                   skelCoordinates(point, 2),
                                                                   skelCoordinates(point, 3));
            }
        }

    }


    // TODO support ROI
    // TODO output data (probably 'SkeletonDictionary')
    inline void getSkeletonNodeAssignments(const std::string & segmentationPath,
                                           const std::string & skeletonTopFolder,
                                           const std::vector<size_t> & skeletonIds,
                                           const int numberOfThreads,
                                           SkeletonDictionary & out) {
        // threadpool
        parallel::ThreadPool threadpool(numberOfThreads);

        // std::cout << "AAA" << std::endl;
        // group the skeleton parts by the chunks of the segmentation
        // dataset they fall into
        SkeletonBlockStorage skeletonsToBlocks;
        std::vector<size_t> nonEmptyChunks;
        groupSkeletonBlocks(segmentationPath,
                            skeletonTopFolder,
                            skeletonIds,
                            threadpool, skeletonsToBlocks,
                            nonEmptyChunks);

        // std::cout << "BBB" << std::endl;
        // extract the node assignments for all blocks in parallel
        const size_t nThreads = threadpool.nThreads();
        std::vector<SkeletonDictionary> perThreadData(nThreads);

        // std::cout << "CCC" << std::endl;
        const size_t nChunks = nonEmptyChunks.size();
        parallel::parallel_foreach(threadpool, nChunks, [&](const int tid, const size_t chunkIndex){
            const size_t chunkId = nonEmptyChunks[chunkIndex];
            auto & threadData = perThreadData[tid];
            extractNodeAssignmentsForBlock(segmentationPath, chunkId, skeletonsToBlocks[chunkId], threadData);
        });

        // initialize the skeletons
        for(const size_t skeletonId : skeletonIds) {
            out.insert(std::make_pair(skeletonId, SkeletonNodeAssignment()));
        }

        // std::cout << "DDD" << std::endl;
        // merge the node assignments for all skeletons
        const size_t nSkeletons = skeletonIds.size();
        parallel::parallel_foreach(threadpool, nSkeletons, [&](const int tid, const size_t skeletonIndex) {
            const size_t skeletonId = skeletonIds[skeletonIndex];
            auto & nodeAssignment = out[skeletonId];
            // iterate over all threads and
            for(size_t t = 0; t < nThreads; ++t) {
                const auto & threadData = perThreadData[t];
                auto skelIt = threadData.find(skeletonId);
                if(skelIt == threadData.end()) {
                    continue;
                }
                // could use merge implementation if we had C++ 17 ;(
                nodeAssignment.insert(skelIt->second.begin(), skelIt->second.end());
            }
        });
    }

}
}
