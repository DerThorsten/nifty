#pragma once
#ifndef NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX
#define NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX

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

        ;
    }
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX
