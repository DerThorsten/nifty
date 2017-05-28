#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/graph/shortest_path_dijkstra.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace py = pybind11;

// TODO we need a grid graph to use this for false merge stuff
// -> best to implement a wrapper for some grid graph in a different module

namespace nifty{
namespace graph{

    typedef std::vector<int64_t> NodePath;
    typedef std::vector<int64_t> EdgePath;

    template <typename SP_TYPE>
    NodePath pathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const int64_t target) {
        const auto & predecessors = sp.predecessors();
        const int64_t invalidNode = -1;
        NodePath path;
        int64_t next = target;
        while(next != source) {
            path.push_back(next);
            next = predecessors[next];
            // invalid node -> there is no path between target and source
            // we return an empty path
            if(next == invalidNode) {
                NodePath emptyPath;
                return emptyPath;
            }
        }
        path.push_back(source);
        return path;
    }
    
    template <typename SP_TYPE>
    EdgePath edgePathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const int64_t target) {

        const auto & predecessors = sp.predecessors();
        const int64_t invalidNode = -1;
        const auto & graph = sp.graph();

        EdgePath path;
        int64_t last = target;
        int64_t edge, next;
        while(next != source) {

            next = predecessors[next];
            // invalid node -> there is no path between target and source
            // we return an empty path
            if(next == invalidNode) {
                EdgePath emptyPath;
                return emptyPath;
            }
            edge = graph.findEdge(last, next);
            path.push_back(edge);
            last = next;
        }
        return path;
    }
    
    template <typename SP_TYPE>
    std::vector<NodePath> pathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const std::vector<int64_t> & targets) {
        std::vector<NodePath> paths;
        for(auto trgt : targets)
            paths.push_back(pathsFromPredecessors(sp, source, trgt));
        return paths;
    }
    
    template <typename SP_TYPE>
    std::vector<EdgePath> edgePathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const std::vector<int64_t> & targets) {
        std::vector<EdgePath> paths;
        for(auto trgt : targets)
            paths.push_back(edgePathsFromPredecessors(sp, source, trgt));
        return paths;
    }

    template<typename WEIGHT_TYPE>
    void exportShortestPathDijkstraT(py::module & graphModule) {
        
        typedef UndirectedGraph<> GraphType;
        typedef WEIGHT_TYPE WeightType;
        typedef ShortestPathDijkstra<GraphType, WeightType> ShortestPathType;
        typedef std::vector<WeightType> EdgeWeightsType;
        
        const auto clsName = std::string("ShortestPathDijkstra");
        auto shortestPathCls = py::class_<ShortestPathType>(graphModule, clsName.c_str());

        shortestPathCls
            .def(py::init<const GraphType &>())
            .def("runSingleSourceSingleTarget", // single source -> single target
                [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source, const int64_t target, const bool returnNodes) {
                    self.runSingleSourceSingleTarget(weights, source, target); 
                    if(returnNodes)
                        return pathsFromPredecessors(self, source, target);
                    else
                        return edgePathsFromPredecessors(self, source, target);
                }
            )
            .def("runSingleSourceMultiTarget", // single source -> multiple targets
                [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source, const std::vector<int64_t> & targets, const bool returnNodes) {
                    self.runSingleSourceMultiTarget(weights, source, targets);
                    if(returnNodes)
                        return pathsFromPredecessors(self, source, targets);
                    else
                        return edgePathsFromPredecessors(self, source, targets);
                }
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
            [](const GraphType & graph, const EdgeWeightsType & edgeWeights, const NodeVector & sources, const NodeVector & targets, const int nThreads) {
                
                std::vector<NodePath> nodePaths(sources.size());
                parallel::ThreadPool threadpool(nThreads);
                
                // initialize a shortest path class for each thread
                std::vector<ShortestPathType> shortestPathThreads( threadpool.nThreads(), ShortestPathType(graph) );
                
                {
                    py::gil_scoped_release allowThreads;
                    parallel::parallel_foreach(threadpool, sources.size(), [&](const int tid, const int ii) {
                        auto & sp = shortestPathThreads[tid];
                        sp.runSingleSourceSingleTarget(edgeWeights, sources[ii], targets[ii]);
                        nodePaths[ii] = pathsFromPredecessors(sp, sources[ii], targets[ii]);
                    });
                }
                return nodePaths;
            },
            py::arg("graph"),py::arg("edgeWeights"),py::arg("sources"),py::arg("targets"),py::arg("nThreads")=-1
        ); 
        
        graphModule.def("shortestPathMultiTargetParallel",
            [](const GraphType & graph, const EdgeWeightsType & edgeWeights, const NodeVector & sources, const std::vector<NodeVector> & targetVectors, const int nThreads) {
                
                std::vector<std::vector<NodePath>> nodePaths(sources.size());
                parallel::ThreadPool threadpool(nThreads);
                
                // initialize a shortest path class for each thread
                std::vector<ShortestPathType> shortestPathThreads( threadpool.nThreads(), ShortestPathType(graph) );
                
                {
                    py::gil_scoped_release allowThreads;
                    parallel::parallel_foreach(threadpool, sources.size(), [&](const int tid, const int ii) {
                        auto & sp = shortestPathThreads[tid];
                        sp.runSingleSourceMultiTarget(edgeWeights, sources[ii], targetVectors[ii]);
                        nodePaths[ii] = pathsFromPredecessors(sp, sources[ii], targetVectors[ii]);
                    });
                }
                return nodePaths;
            },
            py::arg("graph"),py::arg("edgeWeights"),py::arg("sources"),py::arg("targetVectors"),py::arg("nThreads")=-1
        ); 
    }


    void exportShortestPathDijkstra(py::module & graphModule) {
        exportShortestPathDijkstraT<float>(graphModule);
        exportParallelShortestPathT<float>(graphModule);
        // TODO this does not work
        //exportShortestPathDijkstraT<double>(graphModule);
    }

} // namespace graph
} // namespace nifty
