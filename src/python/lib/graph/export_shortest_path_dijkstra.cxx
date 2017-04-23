#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/graph/shortest_path_dijkstra.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace py = pybind11;

// TODO we need a grid graph to use this for false merge stuff
// -> best to implement a wrapper for some grid graph in a different module

namespace nifty{
namespace graph{

    typedef std::vector<int64_t> nodePath;

    // TODO handle cases when no path exists
    template <typename SP_TYPE>
    nodePath pathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const int64_t target) {
        const auto & predecessors = sp.predecessors();
        nodePath path;
        int64_t next = target;
        // TODO what's the invalid key? -> if we hit it we need to break out of the loop and return invalid key
        while(next != source) {
            path.push_back(next);
            next = predecessors[next];
            // invalid node
            //if(next==invalid)
            //  return nodePath(invalid);
        }
        return path;
    }
    
    template <typename SP_TYPE>
    std::vector<nodePath> pathsFromPredecessors(
            const SP_TYPE & sp,
            const int64_t source,
            const std::vector<int64_t> & targets) {
        std::vector<nodePath> paths;
        for(auto trgt : targets)
            paths.push_back(pathsFromPredecessors(sp, source, trgt));
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
            .def(py::init<const GraphType &>()
                //,py::arg_t<GraphType>("graph")
            )
            .def("runSingleSourceSingleTarget", // single source -> single target
                [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source, const int64_t target) {
                    self.runSingleSourceSingleTarget(weights, source, target); 
                    return pathsFromPredecessors(self, source, target);
                }
            )
            .def("runSingleSourceMultiTarget", // single source -> multiple targets
                [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source, const std::vector<int64_t> & targets) {
                    self.runSingleSourceMultiTarget(weights, source, targets); // TODO implement
                    return pathsFromPredecessors(self, source, targets);
                }
            )
            //.def("runSingleSource", // single source -> all targets, don't really see why we would want to expose this
            //    [](ShortestPathType & self, const EdgeWeightsType & weights, const int64_t source) {
            //        self.runSingleSource(weights, source);
            //    }
            //)
        ;
    }
    
    void exportShortestPathDijkstra(py::module & graphModule) {
        exportShortestPathDijkstraT<float>(graphModule);
        //exportShortestPathDijkstraT<double>(graphModule);
    }

} // namespace graph
} // namespace nifty
