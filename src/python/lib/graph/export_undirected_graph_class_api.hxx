#pragma once
#ifndef NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX
#define NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX

#include "../converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{


    template<class G, class ITER>
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
        PyGraphIter( 
            const G & g, 
            py::object gRef
        )   :
            g_(g),
            current_(g.edgesBegin()),
            end_(g.edgesEnd()),
            gRef_(gRef)
        {

        }

        ReturnType next(){
            if(current_ == end_){
                throw py::stop_iteration();
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
        typedef PyGraphIter<G,EdgeIter> PyEdgeIter;
        auto edgeIterClsName = clsName + std::string("EdgeIter");

        py::class_<PyEdgeIter>(graphModule, edgeIterClsName.c_str())
            .def("__iter__", [](PyEdgeIter &it) -> PyEdgeIter& { return it; })
            .def("__next__", &PyEdgeIter::next);
        ;

        cls
            .def("numberOfNodes",&G::numberOfNodes)
            .def("numberOfEdges",&G::numberOfEdges)
            .def("edges", [](py::object g) { return PyEdgeIter(g.cast<const G &>(), g); })
        ;
    }
    


} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_PYTHON_GRAPH_EXPORT_UNDIRECTED_GRAPH_CLASS_API_HXX
