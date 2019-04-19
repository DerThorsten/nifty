#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/graph/shortest_path_dijkstra.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/graph/undirected_grid_graph.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace py = pybind11;

// TODO we need a grid graph to use this for false merge stuff
// -> best to implement a wrapper for some grid graph in a different module

namespace nifty{
namespace graph{

    typedef std::vector<int64_t> Path;

    template <typename SP_TYPE>
    inline void pathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const int64_t target,
            Path & path) {

        path.clear();
        const auto & predecessors = sp.predecessors();
        const int64_t invalidNode = -1;
        int64_t next = target;
        while(next != source) {
            path.push_back(next);
            next = predecessors[next];
            // invalid node -> there is no path between target and source
            // we return an empty path
            if(next == invalidNode) {
                path.clear();
                return;
            }
        }
        path.push_back(source);
    }


    template <typename SP_TYPE>
    inline void edgePathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const int64_t target,
            Path & path) {

        const auto & predecessors = sp.predecessors();
        const int64_t invalidNode = -1;
        const auto & graph = sp.graph();

        path.clear();
        int64_t next = target;
        int64_t last = target;
        int64_t edge;

        while(next != source) {

            next = predecessors[next];
            // invalid node -> there is no path between target and source
            // we return an empty path
            if(next == invalidNode) {
                path.clear();
                return;
            }
            edge = graph.findEdge(last, next);
            path.push_back(edge);
            last = next;
        }
    }

    template <typename SP_TYPE>
    inline void pathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const std::vector<int64_t> & targets,
            std::vector<Path> & paths) {
        paths.clear();
        paths.resize(targets.size());
        for(std::size_t ii = 0; ii < targets.size(); ++ii) {
            pathsFromPredecessors(sp, source, targets[ii], paths[ii]);
        }
    }

    template <typename SP_TYPE>
    inline void edgePathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const std::vector<int64_t> & targets,
            std::vector<Path> & paths) {
        paths.clear();
        paths.resize(targets.size());
        for(std::size_t ii = 0; ii < targets.size(); ++ii) {
            edgePathsFromPredecessors(sp, source, targets[ii], paths[ii]);
        }
    }

    template<typename WEIGHT_TYPE, typename GRAPH_TYPE>
    void exportShortestPathDijkstraT(py::module & graphModule) {

        typedef GRAPH_TYPE GraphType;
        typedef WEIGHT_TYPE WeightType;
        typedef ShortestPathDijkstra<GraphType, WeightType> ShortestPathType;
        typedef std::vector<WeightType> EdgeWeightsType;

        const auto clsName = std::string("ShortestPathDijkstra");
        auto shortestPathCls = py::class_<ShortestPathType>(graphModule, clsName.c_str());

        shortestPathCls
            .def(py::init<const GraphType &>())

            .def("runSingleSourceSingleTarget", // single source -> single target
                [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source, const int64_t target, const bool returnNodes) {

                    Path path;
                    {
                        py::gil_scoped_release allowThreads;
                        self.runSingleSourceSingleTarget(weights, source, target);
                        if(returnNodes)
                            pathsFromPredecessors(self, source, target, path);
                        else
                            edgePathsFromPredecessors(self, source, target, path);
                    }
                    return path;
                },
                py::arg("weights"), py::arg("source"),
                py::arg("target"), py::arg("returnNodes")=true
            )

            .def("runSingleSourceMultiTarget", // single source -> multiple targets
                [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source, const std::vector<int64_t> & targets, const bool returnNodes) {

                std::vector<Path> paths;
                    {
                        py::gil_scoped_release allowThreads;
                        self.runSingleSourceMultiTarget(weights, source, targets);
                        if(returnNodes)
                            pathsFromPredecessors(self, source, targets, paths);
                        else
                            edgePathsFromPredecessors(self, source, targets, paths);
                    }
                    return paths;
                },
                py::arg("weights"), py::arg("source"),
                py::arg("targets"), py::arg("returnNodes")=true
            )
        ;
    }


    template<typename WEIGHT_TYPE>
    void exportParallelShortestPathT(py::module & graphModule) {

        typedef UndirectedGraph<> GraphType;
        typedef WEIGHT_TYPE WeightType;
        typedef ShortestPathDijkstra<GraphType, WeightType> ShortestPathType;
        typedef std::vector<WeightType> EdgeWeightsType;
        typedef std::vector<int64_t> NodeVector;

        graphModule.def("shortestPathSingleTargetParallel",
            [](
                const GraphType & graph,
                const EdgeWeightsType & edgeWeights,
                const NodeVector & sources,
                const NodeVector & targets,
                const bool returnNodes,
                const int numberOfThreads) {

                std::vector<Path> paths(sources.size());

                {
                    py::gil_scoped_release allowThreads;

                    parallel::ThreadPool threadpool(numberOfThreads);
                    // initialize a shortest path class for each thread
                    std::vector<ShortestPathType> shortestPathThreads( threadpool.nThreads(), ShortestPathType(graph) );

                    parallel::parallel_foreach(threadpool, sources.size(), [&](const int tid, const int ii) {
                        auto & sp = shortestPathThreads[tid];
                        sp.runSingleSourceSingleTarget(edgeWeights, sources[ii], targets[ii]);
                        if(returnNodes)
                            pathsFromPredecessors(sp, sources[ii], targets[ii], paths[ii]);
                        else
                            edgePathsFromPredecessors(sp, sources[ii], targets[ii], paths[ii]);
                    });
                }
                return paths;

            },
            py::arg("graph"), py::arg("edgeWeights"),
            py::arg("sources"),py::arg("targets"),
            py::arg("returnNodes")=true, py::arg("numberOfThreads")=-1
        );


        graphModule.def("shortestPathMultiTargetParallel",
            [](
                const GraphType & graph,
                const EdgeWeightsType & edgeWeights,
                const NodeVector & sources,
                const std::vector<NodeVector> & targetVectors,
                const bool returnNodes,
                const int numberOfThreads) {

                std::vector<std::vector<Path>> paths(sources.size());

                {
                    py::gil_scoped_release allowThreads;

                    parallel::ThreadPool threadpool(numberOfThreads);
                    // initialize a shortest path class for each thread
                    std::vector<ShortestPathType> shortestPathThreads(threadpool.nThreads(), ShortestPathType(graph));

                    parallel::parallel_foreach(threadpool, sources.size(), [&](const int tid, const int ii) {
                        auto & sp = shortestPathThreads[tid];
                        sp.runSingleSourceMultiTarget(edgeWeights,
                                                      sources[ii],
                                                      targetVectors[ii]);
                        if(returnNodes)
                            pathsFromPredecessors(sp, sources[ii],
                                                  targetVectors[ii], paths[ii]);
                        else
                            edgePathsFromPredecessors(sp, sources[ii],
                                                      targetVectors[ii], paths[ii]);
                    });
                }
                return paths;
            },
            py::arg("graph"), py::arg("edgeWeights"),
            py::arg("sources"), py::arg("targetVectors"),
            py::arg("returnNodes")=true, py::arg("numberOfThreads")=-1
        );
    }


    void exportShortestPathDijkstra(py::module & graphModule) {
        // for undirected graph
        {
            typedef UndirectedGraph<> GraphType;
            exportShortestPathDijkstraT<float, GraphType>(graphModule);
            exportParallelShortestPathT<float>(graphModule);
        }
        // for grid graph
        {
            // TODO more exports
            typedef UndirectedGridGraph<3, true> GraphType;
            exportShortestPathDijkstraT<float, GraphType>(graphModule);
        }
    }

} // namespace graph
} // namespace nifty
