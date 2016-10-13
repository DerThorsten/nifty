#pragma once
#ifndef NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX
#define NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX

#include "nifty/graph/breadth_first_search.hxx"
#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    template<class G, class ITER, class TAG>
    class PyGraphIter{
    public:
        typedef ITER Iter;
        typedef typename std::iterator_traits<ITER>::value_type ReturnType;
        PyGraphIter( 
            const G & g, 
            py::object gRef,
            const Iter beginIter,
            const Iter endIter
        )   :
            g_(g),
            current_(beginIter),
            end_(endIter),
            gRef_(gRef)
        {
        }

        ReturnType next(){
            if(current_ == end_){
                throw py::stop_iteration();
            }
            else{
            }
            const auto ret = *current_;
            ++current_;
            return ret;
        }


    private:
        const G & g_;
        py::object gRef_;
        Iter current_,end_;

    };

    template<class G, class MAP_TYPE>
    void exportEdgeMap(
        py::module & graphModule,
        const std::string & clsName
    ){

        py::class_<MAP_TYPE>(graphModule, clsName.c_str())
        ;

    }

    template<class G, class MAP_TYPE>
    void exportNodeMap(
        py::module & graphModule,
        const std::string & clsName
    ){

        py::class_<MAP_TYPE>(graphModule, clsName.c_str())
        ;

    }

    template<class G, class CLS_T>
    void exportUndirectedGraphClassAPI(
        py::module & graphModule,
        CLS_T & cls,
        const std::string & clsName
    ){
        
        typedef typename G::EdgeIter EdgeIter;
        typedef PyGraphIter<G,EdgeIter, EdgeTag> PyEdgeIter;
        auto edgeIterClsName = clsName + std::string("EdgeIter");
        py::class_<PyEdgeIter>(graphModule, edgeIterClsName.c_str())
            .def("__iter__", [](PyEdgeIter &it) -> PyEdgeIter& { return it; })
            .def("__next__", &PyEdgeIter::next);
        ;

        typedef typename G::NodeIter NodeIter;
        typedef PyGraphIter<G,NodeIter,NodeTag> PyNodeIter;
        auto nodeIterClsName = clsName + std::string("NodeIter");
        py::class_<PyNodeIter>(graphModule, nodeIterClsName.c_str())
            .def("__iter__", [](PyNodeIter &it) -> PyNodeIter& { return it; })
            .def("__next__", &PyNodeIter::next);
        ;


        typedef typename G:: template EdgeMap<double> EdgeMapFloat64;
        exportEdgeMap<G, EdgeMapFloat64>(graphModule, clsName + std::string("EdgeMapFloat64"));

        typedef typename G:: template NodeMap<double> NodeMapFloat64;
        exportEdgeMap<G, NodeMapFloat64>(graphModule, clsName + std::string("NodeMapFloat64"));

        cls
            .def_property_readonly("numberOfNodes",&G::numberOfNodes)
            .def_property_readonly("numberOfEdges",&G::numberOfEdges)
            .def_property_readonly("nodeIdUpperBound",&G::nodeIdUpperBound)
            .def_property_readonly("edgeIdUpperBound",&G::edgeIdUpperBound)

            .def("findEdge",[](const G & self, std::pair<uint64_t, uint64_t> uv){
                return self.findEdge(uv.first, uv.second);
            })
            .def("findEdge",&G::findEdge)
            .def("u",&G::u)
            .def("v",&G::v)
            .def("uv",&G::uv)
            .def("edges", [](py::object g) { 
                const auto & gg = g.cast<const G &>();
                return PyEdgeIter(gg,g,gg.edgesBegin(),gg.edgesEnd()); 
            })
            .def("nodes", [](py::object g) { 
                const auto & gg = g.cast<const G &>();
                return PyNodeIter(gg,g,gg.nodesBegin(),gg.nodesEnd()); 
            })

            .def("__str__",
                [](const G & g) {
                    std::stringstream ss;
                    ss<<"#Nodes "<<g.numberOfNodes()<<" #Edges "<<g.numberOfEdges();
                    return ss.str();
                }
            )
            .def("__repr__",
                [](const G & g) {
                    std::stringstream ss;
                    auto first = true;
                    for(auto edge : g.edges()){
                        if(first){
                            first = false;
                            ss<<g.u(edge)<<"-"<<g.v(edge);
                        }
                        else
                            ss<<"\n"<<g.u(edge)<<"-"<<g.v(edge);
                    }
                    return ss.str();
                }
            )

            .def("bfsNodes",[](const G & g, const uint64_t maxDistance){

                typedef std::tuple<uint64_t,uint64_t, uint64_t> TupleType;
                std::vector<TupleType > data;


                BreadthFirstSearch<G> bfs(g);
                g.forEachNode([&](const uint64_t sourceNode){
                    bfs.graphNeighbourhood(sourceNode, maxDistance, [&](const uint64_t targetNode, const uint64_t dist){
                        
                        if(sourceNode < targetNode){
                            data.push_back(TupleType(sourceNode, targetNode, dist));
                        }
                    });
                });

                typedef nifty::marray::PyView<uint64_t> Array;
                std::pair< Array, Array > res;

                auto & pyBfsNodes = res.first;
                auto & pyDists = res.second;

                pyBfsNodes.reshapeIfEmpty({data.size(), size_t(2)});
                pyDists.reshapeIfEmpty({data.size()});



                auto c = 0;
                for(const auto & d : data){
                    pyBfsNodes(c,0) = std::get<0>(d);
                    pyBfsNodes(c,1) = std::get<1>(d);
                    pyDists(c) = std::get<2>(d);
                    ++c;
                }

                return res;
            })

        ;
    }
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX
