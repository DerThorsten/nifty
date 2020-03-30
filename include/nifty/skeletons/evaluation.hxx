#include "boost/geometry.hpp"
#include "boost/geometry/index/rtree.hpp"
#include "boost/serialization/map.hpp"
#include "boost/serialization/unordered_map.hpp"
// can't build with boost serialization
//#include "boost/archive/binary_iarchive.hpp"
//#include "boost/archive/binary_oarchive.hpp"

#include "nifty/z5/z5.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;


namespace nifty {
namespace skeletons {

    // TODO support ROI -> not sure if this should be done in here
    // or when extracting skeletons (but even then, we at least need
    // a coordinate offset if we don't want to copy labels)
    class SkeletonMetrics {

    // typedefs
    public:
        // store coordinates of a single (sub-) skeleton
        typedef xt::xtensor<std::size_t, 2> CoordinateArray;
        typedef typename CoordinateArray::shape_type ArrayShape;
        // store tmp coordinates as vector
        typedef std::vector<std::size_t> CoordinateVector;
        // store (sub-) skeleton id to coordinates assignment
        typedef std::map<std::size_t, CoordinateArray> SkeletonStorage;
        // store (sub-) skeletons for all blocks
        typedef std::map<std::size_t, SkeletonStorage> SkeletonBlockStorage;
        // assignmet of skeleton nodes to segmentation nodes
        typedef std::unordered_map<std::size_t, std::size_t> SkeletonNodeAssignment;
        typedef std::map<std::size_t, SkeletonNodeAssignment> SkeletonDictionary;

        // typefs for boost r-tree
        typedef bg::model::point<std::size_t, 3, bg::cs::cartesian> Point;
        typedef bg::model::box<Point> Box;
        typedef std::pair<Point, std::size_t> TreeValue;

        // stores the distances to boundary pixels for all nodes of a given skeleton
        typedef std::unordered_map<std::size_t, std::vector<double>> NodeDistanceStatistics;

        // stores the node distance statistics for all skeletons
        typedef std::unordered_map<std::size_t, NodeDistanceStatistics> SkeletonDistanceStatistics;


    // private members
    private:
        std::string segmentationPath_;
        std::string segmentationKey_;
        std::string skeletonPath_;
        std::string skeletonPrefix_;
        // we might consider holding this as a reference,
        // but for now this is so little data that a copy doesn't matter
        std::vector<std::size_t> skeletonIds_;
        SkeletonDictionary skeletonDict_;
        // TODO node labelngs


    // API
    public:

        // needs boost serialization
        /*
        // constructor from serialization
        SkeletonMetrics(const std::string & segmentationPath,
                        const std::string & skeletonTopFolder,
                        const std::vector<std::size_t> & skeletonIds,
                        const std::string & dictSerialization) : segmentationPath_(segmentationPath),
                                                                 skeletonTopFolder_(skeletonTopFolder),
                                                                 skeletonIds_(skeletonIds){
            deserialize(dictSerialization);
        }
        */

        // constructor from data
        SkeletonMetrics(const std::string & segmentationPath,
                        const std::string & segmentationKey,
                        const std::string & skeletonPath,
                        const std::string & skeletonPrefix,
                        const std::vector<std::size_t> & skeletonIds,
                        const int numberOfThreads) : segmentationPath_(segmentationPath),
                                                     segmentationKey_(segmentationKey),
                                                     skeletonPath_(skeletonPath),
                                                     skeletonPrefix_(skeletonPrefix),
                                                     skeletonIds_(skeletonIds){
            init(numberOfThreads);
        }

        // expose node assignement
        const SkeletonDictionary & getNodeAssignments() const {
            return skeletonDict_;
        }

        // get edges that contain splits
        void getSplitEdges(std::map<std::size_t, std::vector<bool>> &, const int numberOfThreads) const;
        // get edges that contain merges
        void getMergeEdges(std::map<std::size_t, std::vector<bool>> &,
                           std::map<std::size_t, std::size_t> &,
                           const int numberOfThreads) const;

        // compute the split, merge and summary (google) score
        void computeSplitScores(std::map<std::size_t, double> &, const int) const;
        void computeExplicitMergeScores(std::map<std::size_t, double> &,
                                        std::map<std::size_t, std::size_t> &,
                                        const int) const;
        void computeGoogleScore(double &, double &, double &, std::size_t &, const int) const;

        // compute the split run-length
        void computeSplitRunlengths(const std::array<double, 3> &,
                                    std::map<std::size_t, double> &,
                                    std::map<std::size_t, std::map<std::size_t, double>> &,
                                    const std::size_t) const;
        // compute the explicit merges
        void computeExplicitMerges(std::map<std::size_t, std::vector<std::size_t>> &, const int) const;
        // compute the heuristic merges
        void computeHeuristicMerges(const std::array<double, 3> &,
                                    const double,
                                    std::map<std::size_t, std::vector<std::size_t>> &,
                                    const int) const;

        // compute the distance statistics over all
        void computeDistanceStatistics(const std::array<double, 3> &,
                                       SkeletonDistanceStatistics &,
                                       const int) const;

        // merge nodes belonging to false splits
        void mergeFalseSplitNodes(std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>> &,
                                  const int) const;

        // for each skeleton with explicit merge(s), find the nodes that are assigned to
        // the label id(s) that contain the merge
        void getNodesInFalseMergeLabels(std::map<std::size_t, std::vector<std::size_t>> &, const int) const;

        // we can't build with boost::serialization right now
        // best would be to reimplement this
        /*
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
        */

        // group skeleton to blocks (= chunks of the segmentation)
        void groupSkeletonBlocks(SkeletonBlockStorage &, std::vector<std::size_t> &, parallel::ThreadPool &);

    // private methods
    private:
        // initialize the metrics class from data
        void init(const std::size_t numberOfThreads);
        // extract the node assignment for a single block
        void extractNodeAssignmentsForBlock(const std::size_t, const SkeletonStorage & skeletons, SkeletonDictionary &);
        // build r-trees for merge heuristics
        template<class TREE>
        void buildRTrees(const std::array<double, 3> &,
                         std::unordered_map<std::size_t, TREE> &,
                         parallel::ThreadPool &) const;
        //
        void getSkeletonsToLabel(std::unordered_map<std::size_t, std::set<std::size_t>> &,
                                 std::vector<std::size_t> &,
                                 parallel::ThreadPool &p) const;
        //
        void getLabelsWithoutExplicitMerge(std::unordered_map<std::size_t, std::size_t> &,
                                           parallel::ThreadPool &) const;
        //
        void mapLabelsToSkeletons(std::unordered_map<std::size_t, std::vector<std::size_t>> &, 
                                  parallel::ThreadPool &) const;
    };


    void SkeletonMetrics::init(const std::size_t numberOfThreads) {

        parallel::ThreadPool tp(numberOfThreads);

        // group the skeleton parts by the chunks of the segmentation
        // dataset they fall into
        SkeletonBlockStorage skeletonsToBlocks;
        std::vector<std::size_t> nonEmptyChunks;
        groupSkeletonBlocks(skeletonsToBlocks, nonEmptyChunks, tp);

        // extract the node assignments for all blocks in parallel
        const std::size_t nThreads = tp.nThreads();
        std::vector<SkeletonDictionary> perThreadData(nThreads);

        const std::size_t nChunks = nonEmptyChunks.size();
        parallel::parallel_foreach(tp, nChunks, [&](const int tid, const std::size_t chunkIndex){
            const std::size_t chunkId = nonEmptyChunks[chunkIndex];
            auto & threadData = perThreadData[tid];
            extractNodeAssignmentsForBlock(chunkId, skeletonsToBlocks[chunkId], threadData);
        });

        // initialize the skeleton dictionary
        for(const std::size_t skeletonId : skeletonIds_) {
            skeletonDict_.insert(std::make_pair(skeletonId, SkeletonNodeAssignment()));
        }

        // merge the node assignments for all skeletons
        const std::size_t nSkeletons = skeletonIds_.size();
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tid, const std::size_t skeletonIndex) {
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & nodeAssignment = skeletonDict_[skeletonId];
            // iterate over all threads and
            for(std::size_t t = 0; t < nThreads; ++t) {
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
                                              std::vector<std::size_t> & nonEmptyChunks,
                                              parallel::ThreadPool & tp) {

        std::vector<std::size_t> zeroCoord = {0, 0};

        // open the segmentation dataset and get the shape and chunks
        const z5::filesystem::handle::File file(segmentationPath_);
        auto segmentation = z5::openDataset(file, segmentationKey_);
        const std::size_t nChunks = segmentation->numberOfChunks();

        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<std::size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // go over all skeletons in parallel
        // and extract the sub skeletons for each block
        const std::size_t nSkeletons = skeletonIds_.size();
        const std::size_t nThreads = tp.nThreads();

        // temporary block data for the threads
        std::vector<SkeletonBlockStorage> perThreadData(nThreads);

        const auto & chunking = segmentation->chunking();
        const z5::filesystem::handle::File skelFile(skeletonPath_);

        // go over all skeletons in parallel and extract the parts overlapping with chunks
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            // open the coordinate dataset for this particular skeleton
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            const std::string skeletonKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/coordinates";
            auto coordinateSet = z5::openDataset(skelFile, skeletonKey);
            const std::size_t nPoints = coordinateSet->shape(0);
            // load the coordinate data
            ArrayShape coordShape = {nPoints, coordinateSet->shape(1)};
            CoordinateArray coords(coordShape);
            z5::multiarray::readSubarray<uint64_t>(coordinateSet, coords, zeroCoord.begin());
            // loop over all coordinates, and find the corresponding chunk
            std::map<std::size_t, std::vector<CoordinateVector>> blocksToCoordinates; // maping of coordinates to chunks
            for(std::size_t point = 0; point < nPoints; ++point) {
                // we assume to have four entries per coordinate, the first is the skeleton node index,
                // the rest are the coordinates

                CoordinateVector coordinate = {coords(point, 1), coords(point, 2), coords(point, 3)};
                // get the indices of this chunk and concert them to a flat index
                std::vector<std::size_t> chunkIds;
                chunking.coordinateToBlockCoordinate(coordinate, chunkIds);
                std::size_t chunkId = 0;
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
                const std::size_t chunkId = elem.first;

                // go from flat chunk index to chunk indices
                std::vector<std::size_t> chunkIds(3);
                std::size_t tmpIdx = chunkId;
                for(unsigned dim = 0; dim < 3; ++dim) {
                    chunkIds[dim] = tmpIdx / chunkStrides[dim];
                    tmpIdx -= chunkIds[dim] * chunkStrides[dim];
                }

                // find the offset of this chunk
                std::vector<std::size_t> chunkOffset;
                segmentation->getChunkOffset(chunkIds, chunkOffset);

                auto threadIt = threadData.find(chunkId);
                if(threadIt == threadData.end()) {
                    threadIt = threadData.insert(threadIt,
                                                 std::make_pair(chunkId, SkeletonStorage()));
                }

                // crop and copy coordinates
                ArrayShape coordShape = {elem.second.size(), 4};
                CoordinateArray coordArray(coordShape);
                std::size_t ii = 0;
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
        std::set<std::size_t> nonEmptyChunkSet;
        for(std::size_t t = 0; t < nThreads; ++t) {
            auto & threadData = perThreadData[t];
            for(auto it = threadData.begin(); it != threadData.end(); ++it) {
                nonEmptyChunkSet.insert(it->first);
            }
        }
        const std::size_t nNonEmpty = nonEmptyChunkSet.size();
        nonEmptyChunks.resize(nNonEmpty);
        std::copy(nonEmptyChunkSet.begin(), nonEmptyChunkSet.end(), nonEmptyChunks.begin());

        // create empty entries for all non-empty chunks
        for(const std::size_t chunkId : nonEmptyChunks) {
            out.insert(std::make_pair(chunkId, SkeletonStorage()));
        }

        // go over all non-empty blocks in parallel and assemble all the sub-skeletons
        parallel::parallel_foreach(tp, nNonEmpty, [&](const int tId, const std::size_t nonEmptyIndex){
            const std::size_t chunkId = nonEmptyChunks[nonEmptyIndex];
            auto & outChunk = out[chunkId];

            // go over all the thread data and insert sub-skeletons of this chunk
            for(std::size_t t = 0; t < nThreads; ++t) {
                auto & threadData = perThreadData[t];
                auto threadIt = threadData.find(chunkId);
                if(threadIt == threadData.end()) {
                    continue;
                }
                outChunk.insert(threadIt->second.begin(), threadIt->second.end());
            }
        });
    }


    inline void SkeletonMetrics::extractNodeAssignmentsForBlock(const std::size_t chunkId,
                                                                const SkeletonStorage & skeletons,
                                                                SkeletonDictionary & out) {
        // open the segmentation dataset and get the shape and chunks
        const z5::filesystem::handle::File file(segmentationPath_);
        auto segmentation = z5::openDataset(file, segmentationKey_);
        const std::size_t nChunks = segmentation->numberOfChunks();

        // we could do all that outside of the function only once,
        // but it shouldn't matter at all for performance

        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<std::size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // go from the chunk id to chunk index vector
        std::vector<std::size_t> chunkIds(3);
        std::size_t tmpIdx = chunkId;
        for(unsigned dim = 0; dim < 3; ++dim) {
            chunkIds[dim] = tmpIdx / chunkStrides[dim];
            tmpIdx -= chunkIds[dim] * chunkStrides[dim];
        }

        // load the chunk data
        std::vector<std::size_t> chunkOffset;
        segmentation->getChunkOffset(chunkIds, chunkOffset);
        std::vector<std::size_t> chunkShape;
        segmentation->getChunkShape(chunkIds, chunkShape);

        typedef typename xt::xtensor<uint64_t, 3> ::shape_type LabelsShape;
        LabelsShape labelsShape = {chunkShape[0], chunkShape[1], chunkShape[2]};
        xt::xtensor<uint64_t, 3> labels(labelsShape);
        z5::multiarray::readSubarray<uint64_t>(segmentation, labels, chunkOffset.begin());

        // iterate over all the skeletons in this chunks and extract the node assignment
        for(const auto & skel : skeletons) {
            const std::size_t skeletonId = skel.first;

            // check if the skeleton dictionary (out) already has this skeleton id
            auto skelIt = out.find(skeletonId);
            if(skelIt == out.end()) {
                skelIt = out.insert(skelIt, std::make_pair(skeletonId, SkeletonNodeAssignment()));
            }
            auto & nodeAssignment = skelIt->second;

            const auto & skelCoordinates = skel.second;
            const std::size_t nPoints = skelCoordinates.shape()[0];
            for(std::size_t point = 0; point < nPoints; ++point) {
                nodeAssignment[skelCoordinates(point, 0)] = labels(skelCoordinates(point, 1),
                                                                   skelCoordinates(point, 2),
                                                                   skelCoordinates(point, 3));
            }
        }
    }


    void SkeletonMetrics::getSkeletonsToLabel(std::unordered_map<std::size_t, std::set<std::size_t>> & labelsPerSkeleton,
                                              std::vector<std::size_t> & labels,
                                              parallel::ThreadPool & tp) const {

        // initialize the skeleton to label data
        for(auto skelId : skeletonIds_) {
            labelsPerSkeleton[skelId] = std::set<std::size_t>();
        }

        // find the unique labels for each skeleton
        const std::size_t nSkeletons = skeletonIds_.size();
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & skelLabels = labelsPerSkeleton[skeletonId];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);
            for(const auto & assignment : nodeAssignments) {
                skelLabels.insert(assignment.second);
            }
        });

        // find the unique labels
        std::set<std::size_t> labelsTmp;
        for(const auto & skelLabels : labelsPerSkeleton) {
            // would be nice to have CPP 17 with optimized merge
            labelsTmp.insert(skelLabels.second.begin(), skelLabels.second.end());
        }
        labels.resize(labelsTmp.size());
        std::copy(labelsTmp.begin(), labelsTmp.end(), labels.begin());
    }


    void SkeletonMetrics::getSplitEdges(std::map<std::size_t, std::vector<bool>> & splitEdges, const int numberOfThreads) const {

        parallel::ThreadPool tp(numberOfThreads);
        std::vector<std::size_t> zeroCoord = {0, 0};

        // prepare the output data
        const std::size_t nSkeletons = skeletonIds_.size();
        splitEdges.clear();
        for(const std::size_t skelId : skeletonIds_) {
            splitEdges[skelId] = std::vector<bool>();
        }

        const z5::filesystem::handle::File skelFile(skeletonPath_);
        // extract the split scores in parallel
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);
            auto & splitEdge = splitEdges[skeletonId];

            // load the skeleton edges
            const std::string skeletonKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/edges";
            auto edgeSet = z5::openDataset(skelFile, skeletonKey);
            const std::size_t nEdges = edgeSet->shape(0);

            // load the edge data
            ArrayShape coordShape = {nEdges, edgeSet->shape(1)};
            CoordinateArray edges(coordShape);
            z5::multiarray::readSubarray<int64_t>(edgeSet, edges, zeroCoord.begin());

            // to find the split score, iterate over the edges
            // the best split score is one -> only add up edges that connect the same
            // segmentation ids
            for(std::size_t edgeId = 0; edgeId < nEdges; ++edgeId) {
                const int64_t skelA = edges(edgeId, 0);
                const int64_t skelB = edges(edgeId, 1);

                // check for invalid edges
                if(skelA == -1 || skelB == -1) {
                    continue;
                }

                // check if parent is not in the nodes we have
                // (this might happen for extracted subvolumes)
                auto nodeIt = nodeAssignments.find(skelB);
                if(nodeIt == nodeAssignments.end()) {
                    continue;
                }
                const std::size_t nodeB = nodeIt->second;
                const std::size_t nodeA = nodeAssignments.at(skelA);

                // check for ignore label
                if(nodeA == 0 || nodeB == 0) {
                    continue;
                }
                splitEdge.push_back(nodeA != nodeB);
            }
        });


    }


    // compute the split score
    void SkeletonMetrics::computeSplitScores(std::map<std::size_t, double> & splitScores, const int numberOfThreads) const {

        parallel::ThreadPool tp(numberOfThreads);
        std::vector<std::size_t> zeroCoord = {0, 0};

        // prepare the output data
        const std::size_t nSkeletons = skeletonIds_.size();
        splitScores.clear();
        for(auto skeletonId : skeletonIds_) {
            splitScores[skeletonId] = 0.;
        }

        std::map<std::size_t, std::vector<bool>> splitEdges;
        getSplitEdges(splitEdges, numberOfThreads);

        // extract the split scores in parallel
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            const auto & splitEdge = splitEdges[skeletonId];
            auto & splitScore = splitScores[skeletonId];

            // we might have no edges, because they were all
            // filtered out
            if(splitEdge.size() == 0) {
                return;
            }

            splitScore = std::accumulate(splitEdge.begin(), splitEdge.end(), 0.);
            // std::cout << "Skeleton: " << skeletonId << " n-splits: " << splitScore << " n-edges: " << splitEdge.size() << std::endl;
            splitScore /= splitEdge.size();

        });
    }


    void SkeletonMetrics::mergeFalseSplitNodes(std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>> & mergeNodes,
                                               const int numberOfThreads) const {

        parallel::ThreadPool tp(numberOfThreads);
        std::vector<std::size_t> zeroCoord = {0, 0};

        // prepare the output data
        const std::size_t nSkeletons = skeletonIds_.size();
        mergeNodes.clear();
        for(const std::size_t skelId : skeletonIds_) {
            mergeNodes[skelId] = std::set<std::pair<std::size_t, std::size_t>>();
        }

        const z5::filesystem::handle::File skelFile(skeletonPath_);
        // extract the split scores in parallel
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);
            auto & mergeNode = mergeNodes[skeletonId];

            // load the skeleton edges
            const std::string skeletonKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/edges";
            auto edgeSet = z5::openDataset(skelFile, skeletonKey);
            const std::size_t nEdges = edgeSet->shape(0);

            // load the edge data
            ArrayShape coordShape = {nEdges, edgeSet->shape(1)};
            CoordinateArray edges(coordShape);
            z5::multiarray::readSubarray<int64_t>(edgeSet, edges, zeroCoord.begin());

            // to find the split score, iterate over the edges
            // the best split score is one -> only add up edges that connect the same
            // segmentation ids
            for(std::size_t edgeId = 0; edgeId < nEdges; ++edgeId) {
                const int64_t skelA = edges(edgeId, 0);
                const int64_t skelB = edges(edgeId, 1);

                // check for invalid edges
                if(skelA == -1 || skelB == -1) {
                    continue;
                }

                // check if parent is not in the nodes we have
                // (this might happen for extracted subvolumes)
                auto nodeIt = nodeAssignments.find(skelB);
                if(nodeIt == nodeAssignments.end()) {
                    continue;
                }
                const std::size_t nodeB = nodeIt->second;
                const std::size_t nodeA = nodeAssignments.at(skelA);

                // check for ignore label
                if(nodeA == 0 || nodeB == 0) {
                    continue;
                }

                if(nodeA != nodeB) {
                    mergeNode.emplace(nodeA, nodeB);
                }
            }
        });

    }


    void SkeletonMetrics::computeSplitRunlengths(const std::array<double, 3> & resolution,
                                                 std::map<std::size_t, double> & skeletonRunlens,
                                                 std::map<std::size_t, std::map<std::size_t, double>> & fragmentRunlens,
                                                 const std::size_t numberOfThreads) const {
        parallel::ThreadPool tp(numberOfThreads);
        std::vector<std::size_t> zeroCoord = {0, 0};

        // initialize the outputs
        const std::size_t nSkeletons = skeletonIds_.size();
        skeletonRunlens.clear();
        fragmentRunlens.clear();
        for(auto skeletonId : skeletonIds_) {
            skeletonRunlens[skeletonId] = 0.;
            fragmentRunlens[skeletonId] = std::map<std::size_t, double>();
        }

        const z5::filesystem::handle::File skelFile(skeletonPath_);
        // iterate over the skelton ids in parallel
        // and extract the runlengths for the skeltons and the
        // fragments that have overlap with the skeleton
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex) {
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & runlen = skeletonRunlens[skeletonId];
            auto & fragLens = fragmentRunlens[skeletonId];
            const auto & nodeAssignments = skeletonDict_.at(skeletonId);

            // load the coordinates
            const std::string coordKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/coordinates";
            auto coordinateSet = z5::openDataset(skelFile, coordKey);

            const std::size_t nPoints = coordinateSet->shape(0);
            ArrayShape coordShape = {nPoints, coordinateSet->shape(1)};
            CoordinateArray coords(coordShape);
            z5::multiarray::readSubarray<uint64_t>(coordinateSet, coords, zeroCoord.begin());

            // get mapping from node-id to (dense) coordinate index
            std::unordered_map<std::size_t, std::size_t> nodeToIndex;
            for(std::size_t ii = 0; ii < nPoints; ++ii) {
                nodeToIndex[coords(ii, 0)] = ii;
            }

            // load the edges
            const std::string edgeKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/edges";
            auto edgeSet = z5::openDataset(skelFile, edgeKey);
            const std::size_t nEdges = edgeSet->shape(0);
            ArrayShape edgeShape = {nEdges, edgeSet->shape(1)};
            CoordinateArray edges(edgeShape);
            z5::multiarray::readSubarray<int64_t>(edgeSet, edges, zeroCoord.begin());

            // iterate over the edges and sum up runlens
            for(std::size_t edgeId = 0; edgeId < nEdges ;++edgeId) {

                const int64_t skelA = edges(edgeId, 0);
                const int64_t skelB = edges(edgeId, 1);

                // check for invalid edges
                if(skelA == -1 || skelB == -1) {
                    continue;
                }

                // check if parent is not in the nodes we have
                // (this might happen for extracted subvolumes)
                auto nodeIt = nodeAssignments.find(skelB);
                if(nodeIt == nodeAssignments.end()) {
                    continue;
                }
                const std::size_t nodeB = nodeIt->second;
                const std::size_t nodeA = nodeAssignments.at(skelA);

                // check for ignore label
                if(nodeA == 0 || nodeB == 0) {
                    continue;
                }

                const std::size_t coordIdA = nodeToIndex[skelA];
                const std::size_t coordIdB = nodeToIndex[skelB];

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


    void SkeletonMetrics::computeExplicitMerges(std::map<std::size_t, std::vector<std::size_t>> & out,
                                                const int numberOfThreads) const {

        parallel::ThreadPool tp(numberOfThreads);
        const std::size_t nThreads = tp.nThreads();

        std::unordered_map<std::size_t, std::set<std::size_t>> labelsPerSkeleton;
        std::vector<std::size_t> labels;
        getSkeletonsToLabel(labelsPerSkeleton, labels, tp);
        const std::size_t nLabels = labels.size();

        // iterate over the unique labels in parallel, and see whether the
        // label is in more than one segment
        std::vector<std::map<std::size_t, std::vector<std::size_t>>> perThreadData(nThreads);

        parallel::parallel_foreach(tp, nLabels, [&](const int tId, const std::size_t labelIndex){
            const std::size_t labelId = labels[labelIndex];

            // we skip the ignore label
            if(labelId == 0) {
                return;
            }

            std::vector<std::size_t> skeletonsWithLabel;
            for(auto skelId : skeletonIds_) {
                const auto & skelLabels = labelsPerSkeleton[skelId];
                if(skelLabels.find(labelId) != skelLabels.end()) {
                    skeletonsWithLabel.push_back(skelId);
                }
            }

            // we only mark a skeleton as having a merge, if more than one label contains it
            if(skeletonsWithLabel.size() > 1) {
                auto & threadData = perThreadData[tId];

                for(const std::size_t skelId : skeletonsWithLabel) {
                    auto skelIt = threadData.find(skelId);
                    if(skelIt == threadData.end()) {
                        threadData.insert(skelIt, std::make_pair(skelId, std::vector<std::size_t>({labelId})));
                    } else {
                        skelIt->second.push_back(labelId);
                    }
                }
            }
        });

        // write data to output
        for(int thread = 0; thread < nThreads; ++thread) {
            auto & threadData = perThreadData[thread];
            // would prefer merge but C++ 17 ...
            for(const auto & elem : threadData) {
                auto outIt = out.find(elem.first);
                if(outIt == out.end()) {
                    out.insert(elem);
                } else {
                    outIt->second.insert(outIt->second.end(), elem.second.begin(), elem.second.end());
                }
            }
        }
    }


    // determine the edges that are part of an (explicit) merge
    void SkeletonMetrics::getMergeEdges(std::map<std::size_t, std::vector<bool>> & mergeEdges,
                                        std::map<std::size_t, std::size_t> & mergePoints,
                                        const int numberOfThreads) const {

        // first, get the labels that are part of a merge
        // merge labels will contain a mapping from skeleton ids
        // to label ids that contain that skeleton
        std::map<std::size_t, std::vector<std::size_t>> mergeLabels;
        computeExplicitMerges(mergeLabels, numberOfThreads);

        parallel::ThreadPool tp(numberOfThreads);
        std::vector<std::size_t> zeroCoord = {0, 0};

        // prepare the output data
        const std::size_t nSkeletons = skeletonIds_.size();
        mergeEdges.clear();
        mergePoints.clear();
        for(const auto & mergePair : mergeLabels) {
            mergeEdges[mergePair.first] = std::vector<bool>();
            mergePoints[mergePair.first] = 0;
        }

        const z5::filesystem::handle::File skelFile(skeletonPath_);
        // extract the merge edges in parallel
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];

            // check if this skeleton has a merge
            // and get the segmentation labels that contain the merge
            auto mergeIt = mergeLabels.find(skeletonId);
            if(mergeIt == mergeLabels.end()) {
                return;
            }
            const auto & labelsWithMerge = mergeIt->second;

            const auto & nodeAssignments = skeletonDict_.at(skeletonId);
            auto & mergeEdge = mergeEdges[skeletonId];

            // load the skeleton edges
            const std::string skeletonKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/edges";
            auto edgeSet = z5::openDataset(skelFile, skeletonKey);
            const std::size_t nEdges = edgeSet->shape(0);

            // load the edge data
            ArrayShape coordShape = {nEdges, edgeSet->shape(1)};
            CoordinateArray edges(coordShape);
            z5::multiarray::readSubarray<int64_t>(edgeSet, edges, zeroCoord.begin());

            // to find the split score, iterate over the edges
            // the best split score is one -> only add up edges that connect the same
            // segmentation ids
            for(std::size_t edgeId = 0; edgeId < nEdges; ++edgeId) {
                const int64_t skelA = edges(edgeId, 0);
                const int64_t skelB = edges(edgeId, 1);

                // check for invalid edges
                if(skelA == -1 || skelB == -1) {
                    continue;
                }

                // check if parent is not in the nodes we have
                // (this might happen for extracted subvolumes)
                auto nodeIt = nodeAssignments.find(skelB);
                if(nodeIt == nodeAssignments.end()) {
                    continue;
                }
                const std::size_t nodeB = nodeIt->second;
                const std::size_t nodeA = nodeAssignments.at(skelA);

                // check for ignore label
                if(nodeA == 0 || nodeB == 0) {
                    continue;
                }

                // check if this edge is a merge edge (i.e. both nodes belong to (the same) merge label)
                // or if it is a merge point (i.e. transition to a merge label)
                bool hasMerge = false;
                if(nodeA == nodeB) {
                    if(std::find(labelsWithMerge.begin(), labelsWithMerge.end(), nodeA) != labelsWithMerge.end()) {
                        hasMerge = true;
                    }
                } else {
                    bool isMergeA = std::find(labelsWithMerge.begin(), labelsWithMerge.end(), nodeA) != labelsWithMerge.end();
                    bool isMergeB = std::find(labelsWithMerge.begin(), labelsWithMerge.end(), nodeB) != labelsWithMerge.end();
                    if(isMergeA || isMergeB) {
                        ++mergePoints[skeletonId];
                    }
                }

                mergeEdge.push_back(hasMerge);
            }
        });
    }


    void SkeletonMetrics::computeExplicitMergeScores(std::map<std::size_t, double> & mergeScores,
                                                     std::map<std::size_t, std::size_t> & mergePoints,
                                                     const int numberOfThreads) const {
        parallel::ThreadPool tp(numberOfThreads);
        std::vector<std::size_t> zeroCoord = {0, 0};

        // prepare the output data
        const std::size_t nSkeletons = skeletonIds_.size();
        mergeScores.clear();
        for(auto skeletonId : skeletonIds_) {
            mergeScores[skeletonId] = 0.;
        }

        std::map<std::size_t, std::vector<bool>> mergeEdges;
        getMergeEdges(mergeEdges, mergePoints, numberOfThreads);

        // extract the merge scores in parallel
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId, const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];

            auto mergeIt = mergeEdges.find(skeletonId);
            if(mergeIt == mergeEdges.end()) {
                return;
            }

            const auto & mergeEdge = mergeIt->second;
            auto & mergeScore = mergeScores[skeletonId];

            mergeScore = std::accumulate(mergeEdge.begin(), mergeEdge.end(), 0.);
            mergeScore /= mergeEdge.size();
        });
    }


    void SkeletonMetrics::computeGoogleScore(double & correctScore, double & splitScore, double & mergeScore,
                                             std::size_t & totalMergePoints, const int numberOfThreads) const {
        std::map<std::size_t, std::vector<bool>> splitEdges;
        getSplitEdges(splitEdges, numberOfThreads);
        std::map<std::size_t, std::vector<bool>> mergeEdges;
        std::map<std::size_t, std::size_t> mergePoints;
        getMergeEdges(mergeEdges, mergePoints, numberOfThreads);

        std::size_t nEdges = 0;
        correctScore = 0.;
        splitScore = 0.;
        mergeScore = 0.;
        totalMergePoints = 0;

        // iterate over all skeletons
        for(const std::size_t skeletonId : skeletonIds_) {
            const auto & splitEdge = splitEdges[skeletonId];
            auto mergeIt = mergeEdges.find(skeletonId);
            if(mergeIt == mergeEdges.end()) {

                for(const bool split : splitEdge) {
                    if(split) {
                        ++splitScore;
                    } else {
                        ++correctScore;
                    }
                }

            } else {
                const auto & mergeEdge = mergeIt->second;

                for(std::size_t edge = 0; edge < splitEdge.size(); ++edge) {
                    if(splitEdge[edge]) {
                        ++splitScore;
                    } else if(mergeEdge[edge]) {
                        ++mergeScore;
                    } else {
                        ++correctScore;
                    }
                }
                totalMergePoints += mergePoints[skeletonId];

            }

            nEdges += splitEdge.size();
        }

        correctScore /= nEdges;
        splitScore /= nEdges;
        mergeScore /= nEdges;
    }


    template<class TREE>
    void SkeletonMetrics::buildRTrees(const std::array<double, 3> & resolution,
                                      std::unordered_map<std::size_t, TREE> & out,
                                      parallel::ThreadPool & tp) const {
        std::vector<std::size_t> zeroCoord = {0, 0};
        const std::size_t nSkeletons = skeletonIds_.size();
        for(auto skelId : skeletonIds_) {
            out[skelId] = TREE();
        }

        const z5::filesystem::handle::File skelFile(skeletonPath_);
        parallel::parallel_foreach(tp, nSkeletons, [&](const int tId,
                                                        const std::size_t skeletonIndex){
            const std::size_t skeletonId = skeletonIds_[skeletonIndex];
            auto & tree = out[skeletonId];

            // load the coordinates
            const std::string skeletonKey = skeletonPrefix_ + "/" + std::to_string(skeletonId) + "/coordinates";
            auto coordinateSet = z5::openDataset(skelFile, skeletonKey);
            const std::size_t nPoints = coordinateSet->shape(0);
            ArrayShape coordShape = {nPoints, coordinateSet->shape(1)};
            CoordinateArray coords(coordShape);
            z5::multiarray::readSubarray<uint64_t>(coordinateSet, coords, zeroCoord.begin());

            // insert all the coordinates into the rtrees
            for(std::size_t coordId = 0; coordId < nPoints; ++coordId) {
                // NOTE: first coordinate entry is skeleton node id
                Point p(coords(coordId, 1) * resolution[0],
                        coords(coordId, 2) * resolution[1],
                        coords(coordId, 3) * resolution[2]);
                tree.insert(std::make_pair(p, coords(coordId, 0)));
            }
        });
    }


    // compute the unique labels and a mapping of label to skeleton
    // we only need skeletons that contain just a single label, because the others
    // are marked as false merges already
    void SkeletonMetrics::getLabelsWithoutExplicitMerge(std::unordered_map<std::size_t, std::size_t> & out,
                                                        parallel::ThreadPool & tp) const {
        std::unordered_map<std::size_t, std::set<std::size_t>> labelsPerSkeleton;
        std::vector<std::size_t> labels;
        getSkeletonsToLabel(labelsPerSkeleton, labels, tp);

        // iterate over the unique labels in parallel and find the labels which don't have
        // an explicit merge
        const std::size_t nThreads = tp.nThreads();
        std::vector<std::unordered_map<std::size_t, std::size_t>> perThreadData(nThreads);

        const std::size_t nLabels = labels.size();
        parallel::parallel_foreach(tp, nLabels, [&](const int tId, const std::size_t labelIndex){
            const std::size_t labelId = labels[labelIndex];
            std::vector<std::size_t> skeletonsWithLabel;
            for(auto skelId : skeletonIds_) {
                const auto & skelLabels = labelsPerSkeleton[skelId];
                if(skelLabels.find(labelId) != skelLabels.end()) {
                    skeletonsWithLabel.push_back(skelId);
                }
            }

            // we only write out labels that don't have an explicit merge, i.e. are only contained in a
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


    //
    void SkeletonMetrics::mapLabelsToSkeletons(std::unordered_map<std::size_t, std::vector<std::size_t>> & out,
                                               parallel::ThreadPool & tp) const {
        std::unordered_map<std::size_t, std::set<std::size_t>> labelsPerSkeleton;
        std::vector<std::size_t> labels;
        getSkeletonsToLabel(labelsPerSkeleton, labels, tp);

        // iterate over the unique labels in parallel and find the labels which don't have
        // an explicit merge
        const std::size_t nThreads = tp.nThreads();
        std::vector<std::unordered_map<std::size_t, std::vector<std::size_t>>> perThreadData(nThreads);

        const std::size_t nLabels = labels.size();
        parallel::parallel_foreach(tp, nLabels, [&](const int tId, const std::size_t labelIndex){
            const std::size_t labelId = labels[labelIndex];
            std::vector<std::size_t> skeletonsWithLabel;
            for(auto skelId : skeletonIds_) {
                const auto & skelLabels = labelsPerSkeleton[skelId];
                if(skelLabels.find(labelId) != skelLabels.end()) {
                    skeletonsWithLabel.push_back(skelId);
                }
            }

            // for thread 0, we write directly to the out data
            if(tId == 0) {
                out[labelId] = std::vector<std::size_t>(skeletonsWithLabel.begin(), skeletonsWithLabel.end());
            } else {
                auto & threadData = perThreadData[tId];
                threadData[labelId] = std::vector<std::size_t>(skeletonsWithLabel.begin(), skeletonsWithLabel.end());
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
                                                 std::map<std::size_t, std::vector<std::size_t>> & out,
                                                 const int numberOfThreads) const {
        typedef typename xt::xtensor<uint64_t, 3> ::shape_type LabelsShape;
        parallel::ThreadPool tp(numberOfThreads);
        const std::size_t nThreads = tp.nThreads();

        const std::size_t nSkeletons = skeletonIds_.size();

        // first build the RTrees
        // TODO if this should be a performance issue, try some KDTree impl
        // TODO or try different algorithm parameters (2nd template argument)
        typedef bgi::rtree<TreeValue, bgi::linear<16>> RTree;
        std::unordered_map<std::size_t, RTree> trees;
        buildRTrees(resolution, trees, tp);

        // compute the unique labels and a mapping of label to skeleton
        std::unordered_map<std::size_t, std::size_t> candidateLabels;
        getLabelsWithoutExplicitMerge(candidateLabels, tp);

        // open the segmentation dataset and get the shape and chunks
        const z5::filesystem::handle::File file(segmentationPath_);
        auto segmentation = z5::openDataset(file, segmentationKey_);
        const std::size_t nChunks = segmentation->numberOfChunks();
        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<std::size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // mutex for insertions in the out data and deletion from candidate labels
        std::mutex mut;

        // iterate over all chunks and check for segments that
        // are false merges according to our heuristic
        parallel::parallel_foreach(tp, nChunks, [&](const int tId, const std::size_t chunkId) {
            // go from the chunk id to chunk index vector
            std::vector<std::size_t> chunkIds(3);
            std::size_t tmpIdx = chunkId;
            for(unsigned dim = 0; dim < 3; ++dim) {
                chunkIds[dim] = tmpIdx / chunkStrides[dim];
                tmpIdx -= chunkIds[dim] * chunkStrides[dim];
            }

            // load the chunk data
            std::vector<std::size_t> chunkOffset;
            segmentation->getChunkOffset(chunkIds, chunkOffset);
            std::vector<std::size_t> chunkShape;
            segmentation->getChunkShape(chunkIds, chunkShape);

            LabelsShape labelsShape = {chunkShape[0], chunkShape[1], chunkShape[2]};
            xt::xtensor<uint64_t, 3> labels(labelsShape);
            z5::multiarray::readSubarray<uint64_t>(segmentation, labels, chunkOffset.begin());

            // iterate over all pixels
            for(std::size_t z = 0; z < labelsShape[0]; ++z) {
                for(std::size_t y = 0; y < labelsShape[1]; ++y) {
                    for(std::size_t x = 0; x < labelsShape[2]; ++x) {
                        const uint64_t label = labels(z, y, x);

                        // we skip the ignore label
                        if(label == 0) {
                            continue;
                        }

                        // check if this label is in the candidate labels. if not continue
                        auto candidateIt = candidateLabels.find(label);
                        if(candidateIt == candidateLabels.end()) {
                            continue;
                        }
                        // if this is a candidate label, get skeleton-id check for a tree-point in the vicinity
                        const std::size_t skeletonId = candidateIt->second;
                        const auto & tree = trees[skeletonId];
                        // build the search box around this point
                        Point pQuery(static_cast<std::size_t>((z + chunkOffset[0]) * resolution[0]),
                                     static_cast<std::size_t>((y + chunkOffset[1]) * resolution[1]),
                                     static_cast<std::size_t>((x + chunkOffset[2]) * resolution[2]));
                        Point pMin(static_cast<std::size_t>(std::max(pQuery.get<0>() - maxDistance, 0.)),
                                   static_cast<std::size_t>(std::max(pQuery.get<1>() - maxDistance, 0.)),
                                   static_cast<std::size_t>(std::max(pQuery.get<2>() - maxDistance, 0.)));
                        // getting bigger than shape here does not hurt
                        Point pMax(static_cast<std::size_t>(pQuery.get<0>() + maxDistance),
                                   static_cast<std::size_t>(pQuery.get<1>() + maxDistance),
                                   static_cast<std::size_t>(pQuery.get<2>() + maxDistance));
                        Box boundingBox(pMin, pMax);
                        // see if any skeleton points lie within the maximum radius of this point
                        auto treeIt = tree.qbegin(bgi::within(boundingBox) &&
                                                  bgi::satisfies([&](const TreeValue & v){return bg::distance(v.first, pQuery) < maxDistance;}));
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
                                    out.insert(outIt, std::make_pair(skeletonId, std::vector<std::size_t>({label})));
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


    // compute the distance statistics for all skeleton nodes
    void SkeletonMetrics::computeDistanceStatistics(const std::array<double, 3> & resolution,
                                                    SkeletonMetrics::SkeletonDistanceStatistics & out,
                                                    const int numberOfThreads) const {

        typedef typename xt::xtensor<uint64_t, 3> ::shape_type LabelsShape;
        parallel::ThreadPool tp(numberOfThreads);
        const std::size_t nThreads = tp.nThreads();

        const std::size_t nSkeletons = skeletonIds_.size();

        // first build the RTrees
        // TODO if this should be a performance issue, try some KDTree impl
        // TODO or try different algorithm parameters (2nd template argument)
        typedef bgi::rtree<TreeValue, bgi::linear<16>> RTree;
        std::unordered_map<std::size_t, RTree> trees;
        buildRTrees(resolution, trees, tp);

        // compute the unique labels and a mapping of label to skeleton

        // old impl, only taking skeletons without merge into accout
        // std::unordered_map<std::size_t, std::size_t> candidateLabels;
        // getLabelsWithoutExplicitMerge(candidateLabels, tp);

        // new impl for all skeletons
        std::unordered_map<std::size_t, std::vector<std::size_t>> candidateLabels;
        mapLabelsToSkeletons(candidateLabels, tp);

        // open the segmentation dataset and get the shape and chunks
        const z5::filesystem::handle::File file(segmentationPath_);
        auto segmentation = z5::openDataset(file, segmentationKey_);
        const std::size_t nChunks = segmentation->numberOfChunks();
        // get chunk strides for conversion from n-dim chunk indices to
        // a flat chunk index
        const auto & chunksPerDimension = segmentation->chunksPerDimension();
        std::vector<std::size_t> chunkStrides = {chunksPerDimension[1] * chunksPerDimension[2], chunksPerDimension[2], 1};

        // initialize the output data
        for(const auto skelId : skeletonIds_) {
            out[skelId] = NodeDistanceStatistics();
        }

        // mutex for insertions in the out data
        std::mutex mut;

        // TODO to be completely precise, we would need to check out an overlap of 1 
        // which however makes it necessary to load 4 chunks instead of just 1
        // with the correct definition of boundary, it is even 7 chunks instead of 1 !

        // iterate over all chunks and find the distance of boundary pixels to skeleton nodes
        parallel::parallel_foreach(tp, nChunks, [&](const int tId, const std::size_t chunkId) {
            // go from the chunk id to chunk index vector
            std::vector<std::size_t> chunkIds(3);
            std::size_t tmpIdx = chunkId;
            for(unsigned dim = 0; dim < 3; ++dim) {
                chunkIds[dim] = tmpIdx / chunkStrides[dim];
                tmpIdx -= chunkIds[dim] * chunkStrides[dim];
            }

            // load the chunk data
            std::vector<std::size_t> chunkOffset;
            segmentation->getChunkOffset(chunkIds, chunkOffset);
            std::vector<std::size_t> chunkShape;
            segmentation->getChunkShape(chunkIds, chunkShape);

            LabelsShape labelsShape = {chunkShape[0], chunkShape[1], chunkShape[2]};
            xt::xtensor<uint64_t, 3> labels(labelsShape);
            z5::multiarray::readSubarray<uint64_t>(segmentation, labels, chunkOffset.begin());

            // iterate over all pixels
            for(std::size_t z = 0; z < labelsShape[0]; ++z) {
                for(std::size_t y = 0; y < labelsShape[1]; ++y) {
                    for(std::size_t x = 0; x < labelsShape[2]; ++x) {
                        const uint64_t label = labels(z, y, x);

                        // we skip the ignore label
                        if(label == 0) {
                            continue;
                        }

                        // check if this label is in the candidate labels. if not continue
                        auto candidateIt = candidateLabels.find(label);
                        if(candidateIt == candidateLabels.end()) {
                            continue;
                        }

                        // check if this point is a boundary pixel, if not continue
                        bool isBoundary = false;
                        for(unsigned dim = 0; dim < 3; ++dim) {
                            std::vector<std::size_t> coordinate = {z, y, x};
                            // find boundary pixel to upper coordinate
                            if(coordinate[dim] + 1 < labelsShape[dim]) {
                                coordinate[dim] += 1;
                                const uint64_t otherLabel = labels(coordinate[0], coordinate[1], coordinate[2]);
                                if(otherLabel != label) {
                                    isBoundary = true;
                                    break;
                                }
                            }
                            // find boundary pixel to lower coordinate
                            if(coordinate[dim] - 1 > 0) {
                                coordinate[dim] -= 1;
                                const uint64_t otherLabel = labels(coordinate[0], coordinate[1], coordinate[2]);
                                if(otherLabel != label) {
                                    isBoundary = true;
                                    break;
                                }
                            }
                        }
                        if(!isBoundary) {
                            continue;
                        }

                        // old impl with unambiguous mapping of labels to skeletons

                        /*
                        // if this is a candidate label, get skeleton-id check for a tree-point in the vicinity
                        const std::size_t skeletonId = candidateIt->second;
                        const auto & tree = trees[skeletonId];
                        // make the query point
                        Point pQuery(static_cast<std::size_t>((z + chunkOffset[0]) * resolution[0]),
                                     static_cast<std::size_t>((y + chunkOffset[1]) * resolution[1]),
                                     static_cast<std::size_t>((x + chunkOffset[2]) * resolution[2]));
                        // find the nearest skeleton point to this pixel
                        auto treeIt = tree.qbegin(bgi::nearest(pQuery, 1));
                        const std::size_t skeletonNode = treeIt->second;
                        // find the distacne
                        const double distance = bg::distance(pQuery, treeIt->first);

                        // TODO keep locks here or perThread data and merge output later 
                        // instead ?
                        // insert the distance into the output

                        {
                            std::lock_guard<std::mutex> guard(mut);
                            auto & skelOut = out[skeletonId];
                            auto outIt = skelOut.find(skeletonNode);
                            if(outIt == skelOut.end()) {
                                skelOut.insert(outIt, std::make_pair(skeletonNode, std::vector<double>({distance})));
                            } else {
                                outIt->second.push_back(distance);
                            }
                        }
                        */

                        // new impl, where label can map to more than a single skeleton
                        const auto & skeletonIds = candidateIt->second;
                        for(const std::size_t skeletonId : skeletonIds) {
                            const auto & tree = trees[skeletonId];
                            // make the query point
                            Point pQuery(static_cast<std::size_t>((z + chunkOffset[0]) * resolution[0]),
                                         static_cast<std::size_t>((y + chunkOffset[1]) * resolution[1]),
                                         static_cast<std::size_t>((x + chunkOffset[2]) * resolution[2]));
                            // find the nearest skeleton point to this pixel
                            auto treeIt = tree.qbegin(bgi::nearest(pQuery, 1));
                            const std::size_t skeletonNode = treeIt->second;
                            // find the distacne
                            const double distance = bg::distance(pQuery, treeIt->first);

                            // insert the distance into the output
                            {
                                std::lock_guard<std::mutex> guard(mut);
                                auto & skelOut = out[skeletonId];
                                auto outIt = skelOut.find(skeletonNode);
                                if(outIt == skelOut.end()) {
                                    skelOut.insert(outIt, std::make_pair(skeletonNode, std::vector<double>({distance})));
                                } else {
                                    outIt->second.push_back(distance);
                                }
                            }
                        }
                    }
                }
            }
        });
    }


    void SkeletonMetrics::getNodesInFalseMergeLabels(std::map<std::size_t, std::vector<std::size_t>> & out,
                                                     const int numberOfThreads) const {
        std::map<std::size_t, std::vector<std::size_t>> skeletonsWithExplicitMerge;
        computeExplicitMerges(skeletonsWithExplicitMerge, numberOfThreads);

        for(const auto & skelWithMerge : skeletonsWithExplicitMerge) {
            const std::size_t skelId = skelWithMerge.first;
            const auto & mergeLabelIds = skelWithMerge.second;
            const auto & nodeAssignment = skeletonDict_.at(skelId);

            std::vector<std::size_t> falseMergeNodes;
            for(const auto & nodePair : nodeAssignment) {
                const std::size_t nodeId = nodePair.first;
                const std::size_t labelId = nodePair.second;
                if(std::find(mergeLabelIds.begin(), mergeLabelIds.end(), labelId) != mergeLabelIds.end()) {
                    falseMergeNodes.push_back(nodeId);
                }
            }
            out.insert(std::make_pair(skelId, falseMergeNodes));
        }

    }



}
}
