#include "boost/geometry.hpp"
#include "boost/geometry/index/rtree.hpp"
#include "boost/serialization/map.hpp"
#include "boost/serialization/unordered_map.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"

#include "nifty/z5/z5.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace fs = boost::filesystem;
namespace nifty {
namespace skeletons {

    // TODO support ROI -> not sure if this should be done in here
    // or when extracting skeletons (but even then, we at least need
    // a coordinate offset if we don't want to copy labels)
    class SkeletonMetrics {

    // typedefs
    public:
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

        // typefs for boost r-tree
        typedef bg::model::point<size_t, 3, bg::cs::cartesian> Point;
        typedef bg::model::box<Point> Box;
        typedef std::pair<Point, size_t> TreeValue;


    // private members
    private:
        std::string segmentationPath_;
        std::string skeletonTopFolder_;
        // we might consider holding this as a reference,
        // but for now this is so little data that a copy doesn't matter
        std::vector<size_t> skeletonIds_;
        SkeletonDictionary skeletonDict_;
        // TODO node labelngs


    // API
    public:

        // constructor from serialization
        SkeletonMetrics(const std::string & segmentationPath,
                        const std::string & skeletonTopFolder,
                        const std::vector<size_t> & skeletonIds,
                        const std::string & dictSerialization) : segmentationPath_(segmentationPath),
                                                                 skeletonTopFolder_(skeletonTopFolder),
                                                                 skeletonIds_(skeletonIds){
            deserialize(dictSerialization);
        }

        // constructor from data
        SkeletonMetrics(const std::string & segmentationPath,
                        const std::string & skeletonTopFolder,
                        const std::vector<size_t> & skeletonIds,
                        const int numberOfThreads) : segmentationPath_(segmentationPath),
                                                     skeletonTopFolder_(skeletonTopFolder),
                                                     skeletonIds_(skeletonIds){
            init(numberOfThreads);
        }

        // expose node assignement
        const SkeletonDictionary & getNodeAssignments() const {
            return skeletonDict_;
        }

        // compute the split score
        void computeSplitScores(std::map<size_t, double> &, const size_t) const;

        // compute the split run-length
        void computeSplitRunlengths(const std::array<double, 3> &,
                                    std::map<size_t, double> &,
                                    std::map<size_t, std::map<size_t, double>> &,
                                    const size_t) const;
        // compute the explicit merges
        void computeExplicitMerges(std::map<size_t, std::vector<size_t>> &, const int) const;
        // compute the heuristic merges
        void computeHeuristicMerges(const std::array<double, 3> &,
                                    const double,
                                    std::map<size_t, std::vector<size_t>> &,
                                    const int) const;

        // serialize and deserialize node dictionary with boost::serialization
        void serialize(const std::string & path) const {
            std::ofstream os(path.c_str(), std::ofstream::out | std::ofstream::binary);
            boost::archive::binary_oarchive oarch(os);
            oarch << skeletonDict_;
        }

        void deserialize(const std::string & path) {
            std::ifstream is(path.c_str(), std::ifstream::in | std::ifstream::binary);
            boost::archive::binary_iarchive iarch(is);
            iarch >> skeletonDict_;
        }


    // private methods
    private:
        // initialize the metrics class from data
        void init(const size_t numberOfThreads);
        // group skeleton to blocks (= chunks of the segmentation)
        void groupSkeletonBlocks(SkeletonBlockStorage &, std::vector<size_t> &, parallel::ThreadPool &);
        // extract the node assignment for a single block
        void extractNodeAssignmentsForBlock(const size_t, const SkeletonStorage & skeletons, SkeletonDictionary &);
        // build r-trees for merge heuristics
        template<class TREE>
        void buildRTrees(const std::array<double, 3> &,
                         std::unordered_map<size_t, TREE> &,
                         parallel::ThreadPool &) const;
        //
        void getSkeletonsToLabel(std::unordered_map<size_t, std::set<size_t>> &,
                                 std::vector<size_t> &,
                                 parallel::ThreadPool &p) const;
        //
        void getLabelsWithoutExplicitMerge(std::unordered_map<size_t, size_t> &,
                                           parallel::ThreadPool &) const;
    };


    void SkeletonMetrics::init(const size_t numberOfThreads) {

        parallel::ThreadPool tp(numberOfThreads);

        // group the skeleton parts by the chunks of the segmentation
        // dataset they fall into
        SkeletonBlockStorage skeletonsToBlocks;
        std::vector<size_t> nonEmptyChunks;
        groupSkeletonBlocks(skeletonsToBlocks, nonEmptyChunks, tp);

        // extract the node assignments for all blocks in parallel
        const size_t nThreads = tp.nThreads();
        std::vector<SkeletonDictionary> perThreadData(nThreads);

        const size_t nChunks = nonEmptyChunks.size();
        parallel::parallel_foreach(tp, nChunks, [&](const int tid, const size_t chunkIndex){
            const size_t chunkId = nonEmptyChunks[chunkIndex];
            auto & threadData = perThreadData[tid];
            extractNodeAssignmentsForBlock(chunkId, skeletonsToBlocks[chunkId], threadData);
        });

        // initialize the skeleton dictionary
        for(const size_t skeletonId : skeletonIds_) {
            skeletonDict_.insert(std::make_pair(skeletonId, SkeletonNodeAssignment()));
        }

        // merge the node assignments for all skeletons
        const size_t nSkeletons = skeletonIds_.size();
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tid, const size_t skeletonIndex) {
            const size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & nodeAssignment = skeletonDict_[skeletonId];
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

    // TODO de-spaghettify
    void SkeletonMetrics::groupSkeletonBlocks(SkeletonBlockStorage & out,
                                              std::vector<size_t> & nonEmptyChunks,
                                              parallel::ThreadPool & tp) {

        std::vector<size_t> zeroCoord = {0, 0};

        // open the segmentation dataset and get the shape and chunks
        auto segmentation = z5::openDataset(segmentationPath_);
        const size_t nChunks = segmentation->numberOfChunks();

        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // go over all skeletons in parallel
        // and extract the sub skeletons for each block
        const size_t nSkeletons = skeletonIds_.size();
        const size_t nThreads = tp.nThreads();

        // temporary block data for the threads
        std::vector<SkeletonBlockStorage> perThreadData(nThreads);

        // go over all skeletons in parallel and extract the parts overlapping with chunks
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const size_t skeletonIndex){
            // open the coordinate dataset for this particular skeleton
            const size_t skeletonId = skeletonIds_[skeletonIndex];
            fs::path skeletonPath(skeletonTopFolder_);
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
        parallel::parallel_foreach(tp, nNonEmpty, [&](const int tId, const size_t nonEmptyIndex){
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


    inline void SkeletonMetrics::extractNodeAssignmentsForBlock(const size_t chunkId,
                                                                const SkeletonStorage & skeletons,
                                                                SkeletonDictionary & out) {
        // open the segmentation dataset and get the shape and chunks
        auto segmentation = z5::openDataset(segmentationPath_);
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


    void SkeletonMetrics::getSkeletonsToLabel(std::unordered_map<size_t, std::set<size_t>> & labelsPerSkeleton,
                                              std::vector<size_t> & labels,
                                              parallel::ThreadPool & tp) const {

        // initialize the skeleton to label data
        for(auto skelId : skeletonIds_) {
            labelsPerSkeleton[skelId] = std::set<size_t>();
        }

        // find the unique labels for each skeleton
        const size_t nSkeletons = skeletonIds_.size();
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const size_t skeletonIndex){
            const size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & skelLabels = labelsPerSkeleton[skeletonId];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);
            for(const auto & assignment : nodeAssignments) {
                skelLabels.insert(assignment.second);
            }
        });

        // find the unique labels
        std::set<size_t> labelsTmp;
        for(const auto & skelLabels : labelsPerSkeleton) {
            // would be nice to have CPP 17 with optimized merge
            labelsTmp.insert(skelLabels.second.begin(), skelLabels.second.end());
        }
        labels.resize(labelsTmp.size());
        std::copy(labelsTmp.begin(), labelsTmp.end(), labels.begin());
    }


    // compute the split score
    void SkeletonMetrics::computeSplitScores(std::map<size_t, double> & splitScores, const size_t numberOfThreads) const {

        parallel::ThreadPool tp(numberOfThreads);
        std::vector<size_t> zeroCoord = {0, 0};

        // prepare the output data
        const size_t nSkeletons = skeletonIds_.size();
        splitScores.clear();
        for(auto skeletonId : skeletonIds_) {
            splitScores[skeletonId] = 0.;
        }

        // extract the split scores in parallel
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const size_t skeletonIndex){
            const size_t skeletonId = skeletonIds_[skeletonIndex];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);
            auto & splitScore = splitScores[skeletonId];

            // load the skeleton edges
            fs::path skeletonPath(skeletonTopFolder_);
            skeletonPath /= std::to_string(skeletonId);
            skeletonPath /= "edges";
            auto edgeSet = z5::openDataset(skeletonPath.string());
            const size_t nEdges = edgeSet->shape(0);

            // load the edge data
            ArrayShape coordShape = {nEdges, edgeSet->shape(1)};
            CoordinateArray edges(coordShape);
            z5::multiarray::readSubarray<uint64_t>(edgeSet, edges, zeroCoord.begin());

            // to find the split score, iterate over the edges
            // the best split score is one -> only add up edges that connect the same
            // segmentation ids
            for(size_t edgeId = 0; edgeId < nEdges; ++edgeId) {
                const size_t nodeA = nodeAssignments.at(edges(edgeId, 0));
                const size_t nodeB = nodeAssignments.at(edges(edgeId, 1));
                if(nodeA == nodeB) {
                    splitScore += 1;
                }
            }
            // normalize by the number of edges
            splitScore /= nEdges;
        });
    }


    void SkeletonMetrics::computeSplitRunlengths(const std::array<double, 3> & resolution,
                                                 std::map<size_t, double> & skeletonRunlens,
                                                 std::map<size_t, std::map<size_t, double>> & fragmentRunlens,
                                                 const size_t numberOfThreads) const {
        parallel::ThreadPool tp(numberOfThreads);
        std::vector<size_t> zeroCoord = {0, 0};

        // initialize the outputs
        const size_t nSkeletons = skeletonIds_.size();
        skeletonRunlens.clear();
        fragmentRunlens.clear();
        for(auto skeletonId : skeletonIds_) {
            skeletonRunlens[skeletonId] = 0.;
            fragmentRunlens[skeletonId] = std::map<size_t, double>();
        }

        // iterate over the skelton ids in parallel
        // and extract the runlengths for the skeltons and the
        // fragments that have overlap with the skeleton
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const size_t skeletonIndex) {
            const size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & runlen = skeletonRunlens[skeletonId];
            auto & fragLens = fragmentRunlens[skeletonId];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);

            // path to the skeleton group
            fs::path skeletonPath(skeletonTopFolder_);
            skeletonPath /= std::to_string(skeletonId);

            // load the coordinates
            fs::path coordinatePath(skeletonPath);
            coordinatePath /= "coordinates";
            auto coordinateSet = z5::openDataset(coordinatePath.string());
            const size_t nPoints = coordinateSet->shape(0);
            ArrayShape coordShape = {nPoints, coordinateSet->shape(1)};
            CoordinateArray coords(coordShape);
            z5::multiarray::readSubarray<uint64_t>(coordinateSet, coords, zeroCoord.begin());

            // get mapping from node-id to (dense) coordinate index
            std::unordered_map<size_t, size_t> nodeToIndex;
            for(size_t ii = 0; ii < nPoints; ++ii) {
                nodeToIndex[coords(ii, 0)] = ii;
            }

            // load the edges
            fs::path edgePath(skeletonPath);
            edgePath /= "edges";
            auto edgeSet = z5::openDataset(edgePath.string());
            const size_t nEdges = edgeSet->shape(0);
            ArrayShape edgeShape = {nEdges, edgeSet->shape(1)};
            CoordinateArray edges(edgeShape);
            z5::multiarray::readSubarray<uint64_t>(edgeSet, edges, zeroCoord.begin());

            // iterate over the edges and sum up runlens
            for(size_t edgeId = 0; edgeId < nEdges ;++edgeId) {

                const size_t nodeA = nodeAssignments.at(edges(edgeId, 0));
                const size_t nodeB = nodeAssignments.at(edges(edgeId, 1));
                const size_t coordIdA = nodeToIndex[edges(edgeId, 0)];
                const size_t coordIdB = nodeToIndex[edges(edgeId, 1)];

                // caclulate the length of this edge (= euclidean distacne of nodes)
                double len = 0.;
                double diff = 0.;
                for(unsigned d = 0; d < 3; ++d) {
                    // 0 enrty in coordinates is the node id !
                    diff = (static_cast<int64_t>(coords(coordIdA, d + 1)) - static_cast<int64_t>(coords(coordIdB, d + 1))) * resolution[d];
                    len += diff * diff;
                }
                len = std::sqrt(len);

                // add up the length
                runlen += len;
                if(nodeA == nodeB) {
                    auto fragIt = fragLens.find(nodeA);
                    if(fragIt == fragLens.end()) {
                        fragLens.insert(fragIt, std::make_pair(nodeA, len));
                    }
                    else {
                        fragIt->second += len;
                    }
                }
            }
        });
    }


    void SkeletonMetrics::computeExplicitMerges(std::map<size_t, std::vector<size_t>> & out,
                                                const int numberOfThreads) const {

        parallel::ThreadPool tp(numberOfThreads);
        const size_t nThreads = tp.nThreads();

        std::unordered_map<size_t, std::set<size_t>> labelsPerSkeleton;
        std::vector<size_t> labels;
        getSkeletonsToLabel(labelsPerSkeleton, labels, tp);
        const size_t nLabels = labels.size();

        // iterate over the unique labels in parallel, and see whether the
        // label is in more than one segment
        std::vector<std::map<size_t, std::vector<size_t>>> perThreadData(nThreads);

        parallel::parallel_foreach(tp, nLabels, [&](const int tId, const size_t labelIndex){
            const size_t labelId = labels[labelIndex];
            std::vector<size_t> skeletonsWithLabel;
            for(auto skelId : skeletonIds_) {
                const auto & skelLabels = labelsPerSkeleton[skelId];
                if(skelLabels.find(labelId) != skelLabels.end()) {
                    skeletonsWithLabel.push_back(skelId);
                }
            }

            // we only mark a skeleton as having a merge, if more than one label contains it
            if(skeletonsWithLabel.size() > 1) {
                auto & threadData = perThreadData[tId];

                for(const size_t skelId : skeletonsWithLabel) {
                    auto skelIt = threadData.find(skelId);
                    if(skelIt == threadData.end()) {
                        threadData.insert(skelIt, std::make_pair(skelId, std::vector<size_t>({labelId})));
                    } else {
                        skelIt->second.push_back(labelId);
                    }
                }
            }
        });

        // write data to output
        for(int thread = 0; thread < nThreads; ++thread) {
            auto & threadData = perThreadData[thread];
            // would prefer merge but C++ 17
            out.insert(threadData.begin(), threadData.end());
        }
    }


    template<class TREE>
    void SkeletonMetrics::buildRTrees(const std::array<double, 3> & resolution,
                                      std::unordered_map<size_t, TREE> & out,
                                      parallel::ThreadPool & tp) const {
        std::vector<size_t> zeroCoord = {0, 0};
        const size_t nSkeletons = skeletonIds_.size();
        for(auto skelId : skeletonIds_) {
            out[skelId] = TREE();
        }

        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId,
                                                        const size_t skeletonIndex){
            const size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & tree = out[skeletonId];

            // load the coordinates
            fs::path skeletonPath(skeletonTopFolder_);
            skeletonPath /= std::to_string(skeletonId);
            skeletonPath /= "coordinates";
            auto coordinateSet = z5::openDataset(skeletonPath.string());
            const size_t nPoints = coordinateSet->shape(0);
            ArrayShape coordShape = {nPoints, coordinateSet->shape(1)};
            CoordinateArray coords(coordShape);
            z5::multiarray::readSubarray<uint64_t>(coordinateSet, coords, zeroCoord.begin());

            // insert all the coordinates into the rtrees
            for(size_t coordId = 0; coordId < nPoints; ++coordId) {
                // NOTE: first coordinate entry is skeleton node id
                Point p(coords(coordId, 1) * resolution[0],
                        coords(coordId, 2) * resolution[1],
                        coords(coordId, 3) * resolution[2]);
                tree.insert(p);
            }
        });
    }


    // compute the unique labels and a mapping of label to skeleton
    // we only need skeletons that contain just a single label, because the others
    // are marked as false merges already
    void SkeletonMetrics::getLabelsWithoutExplicitMerge(std::unordered_map<size_t, size_t> & out,
                                                        parallel::ThreadPool & tp) const {
        std::unordered_map<size_t, std::set<size_t>> labelsPerSkeleton;
        std::vector<size_t> labels;
        getSkeletonsToLabel(labelsPerSkeleton, labels, tp);

        // iterate over the unique labels in parallel and find the labels which don't have
        // an explicit merge
        const size_t nThreads = tp.nThreads();
        std::vector<std::unordered_map<size_t, size_t>> perThreadData(nThreads);

        const size_t nLabels = labels.size();
        parallel::parallel_foreach(tp, nLabels, [&](const int tId, const size_t labelIndex){
            const size_t labelId = labels[labelIndex];
            std::vector<size_t> skeletonsWithLabel;
            for(auto skelId : skeletonIds_) {
                const auto & skelLabels = labelsPerSkeleton[skelId];
                if(skelLabels.find(labelId) != skelLabels.end()) {
                    skeletonsWithLabel.push_back(skelId);
                }
            }

            // we only writout labels that don't have an explicit merge, i.e. are only contained in a 
            // single skeleton
            if(skeletonsWithLabel.size() == 1) {
                // for thread 0, we write directly to the out data
                if(tId == 0) {
                    out[labelId] = skeletonsWithLabel[0];
                } else {
                    auto & threadData = perThreadData[tId];
                    threadData[labelId] = skeletonsWithLabel[0];
                }
            }
        });

        // merge the results into the out data (Note that thread 0 is already `out`)
        for(int thread = 1; thread < nThreads; ++thread) {
            auto & threadData = perThreadData[thread];
            // would prefer merge but C++ 17 ... 
            out.insert(threadData.begin(), threadData.end());
        }
    }


    void SkeletonMetrics::computeHeuristicMerges(const std::array<double, 3> & resolution,
                                                 const double maxDistance,
                                                 std::map<size_t, std::vector<size_t>> & out,
                                                 const int numberOfThreads) const {
        typedef typename xt::xtensor<uint64_t, 3> ::shape_type LabelsShape;
        parallel::ThreadPool tp(numberOfThreads);
        const size_t nThreads = tp.nThreads();

        const size_t nSkeletons = skeletonIds_.size();

        // first build the RTrees
        // TODO if this should be a performance issue, try some KDTree impl
        // TODO or try different algorithm parameters (2nd template argument)
        typedef bgi::rtree<Point, bgi::linear<16>> RTree;
        std::unordered_map<size_t, RTree> trees;
        buildRTrees(resolution, trees, tp);

        // compute the unique labels and a mapping of label to skeleton
        std::unordered_map<size_t, size_t> candidateLabels;
        getLabelsWithoutExplicitMerge(candidateLabels, tp);

        // open the segmentation dataset and get the shape and chunks
        auto segmentation = z5::openDataset(segmentationPath_);
        const size_t nChunks = segmentation->numberOfChunks();
        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // mutex for insertions in the out data and deletion from candidate labels
        std::mutex mut;

        // iterate over all chunks and check for segments that
        // are false merges according to our heuristic
        parallel::parallel_foreach(tp, nChunks, [&](const int tId, const size_t chunkId) {
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

            LabelsShape labelsShape = {chunkShape[0], chunkShape[1], chunkShape[2]};
            xt::xtensor<uint64_t, 3> labels(labelsShape);
            z5::multiarray::readSubarray<uint64_t>(segmentation, labels, chunkOffset.begin());

            // iterate over all pixels
            for(size_t z = 0; z < labelsShape[0]; ++z) {
                for(size_t y = 0; y < labelsShape[1]; ++y) {
                    for(size_t x = 0; x < labelsShape[2]; ++x) {
                        const uint64_t label = labels(z, y, x);
                        // check if this label is in the candidate labels. if not continue
                        auto candidateIt = candidateLabels.find(label);
                        if(candidateIt == candidateLabels.end()) {
                            continue;
                        }
                        // if this is a candidate label, get skeleton-id check for a tree-point in the vicinity
                        const size_t skeletonId = candidateIt->second;
                        const auto & tree = trees[skeletonId];
                        // build the search box around this point
                        Point pQuery(static_cast<size_t>((z + chunkOffset[0]) * resolution[0]),
                                     static_cast<size_t>((y + chunkOffset[1]) * resolution[1]),
                                     static_cast<size_t>((x + chunkOffset[2]) * resolution[2]));
                        Point pMin(static_cast<size_t>(std::max(pQuery.get<0>() - maxDistance, 0.)),
                                   static_cast<size_t>(std::max(pQuery.get<1>() - maxDistance, 0.)),
                                   static_cast<size_t>(std::max(pQuery.get<2>() - maxDistance, 0.)));
                        // getting bigger than shape here does not hurt
                        Point pMax(static_cast<size_t>(pQuery.get<0>() + maxDistance),
                                   static_cast<size_t>(pQuery.get<1>() + maxDistance),
                                   static_cast<size_t>(pQuery.get<2>() + maxDistance));
                        Box boundingBox(pMin, pMax);
                        // see if any skeleton points lie within the maximum radius of this point
                        auto treeIt = tree.qbegin(bgi::within(boundingBox) &&
                                                  bgi::satisfies([&](const Point & v){return bg::distance(v, pQuery) < maxDistance;}));
                        const bool inMaxDist = treeIt != tree.qend();
                        // if we are not in the max distance, we mark this label as merge
                        // and remove it from the candidate labels
                        if(!inMaxDist) {
                            std::lock_guard<std::mutex> guard(mut);
                            // in a very unlikely scenario, we might come here when the key
                            // is already deleted from candidateLabels
                            // hence, we only insert in the out data, if we actually delete a label
                            auto deleted = candidateLabels.erase(label);
                            if(deleted) {
                                // insert in output
                                auto outIt = out.find(skeletonId);
                                if(outIt == out.end()) {
                                    out.insert(outIt, std::make_pair(skeletonId, std::vector<size_t>({label})));
                                } else {
                                    outIt->second.push_back(label);
                                }
                            }
                        }
                    }
                }
            }


        });
    }

}
}
