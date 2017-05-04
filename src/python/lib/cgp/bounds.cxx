#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "export_cell_vector.hxx"

#include "nifty/python/converter.hxx"
#include "nifty/cgp/bounds.hxx"

namespace py = pybind11;



typedef std::vector<nifty::cgp::CellBounds<2,0> > VectorCellBounds2_0;
typedef std::vector<nifty::cgp::CellBounds<2,1> > VectorCellBounds2_1;
PYBIND11_MAKE_OPAQUE(VectorCellBounds2_0);
PYBIND11_MAKE_OPAQUE(VectorCellBounds2_1);


namespace nifty{
namespace cgp{


    template<
        size_t DIM, 
        size_t CELL_TYPE,
        class CLS
    >
    void exportCellBoundsT(py::module & m, py::class_<CLS> & pyCls) {
        pyCls
            .def("__len__", &CLS::size)
            .def("__getitem__", [](const CLS & self, uint32_t i){
                return self[i];
            })
            .def("__iter__",[](const CLS & self){
                const auto begin = &(self[0]);

                return py::make_iterator(begin, begin + self.size());
            }, py::keep_alive<0, 1>())
        ;
    }



    void exportBounds2D(py::module & m) {

        
        

        // cell 0 bounds
        {   
            typedef CellBounds<2,0> Cell0Bounds2D;
            typedef CellBoundsVector<2,0> Cells0BoundsVector2D;

            const std::string clsName = std::string("Cell0Bounds2D");
            auto cls = py::class_<Cell0Bounds2D>(m, clsName.c_str());
            exportCellBoundsT<2, 0, Cell0Bounds2D>(m, cls);

            const std::string clsNameVec = std::string("Cell0BoundsVector2D");
            auto clsVec = py::class_<Cells0BoundsVector2D>(m, clsNameVec.c_str());
            clsVec
                .def("__array__",[](const Cells0BoundsVector2D & self){
                    nifty::marray::PyView<uint32_t> ret({size_t(self.size()),size_t(4)});
                    for(size_t ci=0 ;ci<self.size(); ++ci){
                        for(size_t i=0; i<self[ci].size(); ++i){
                            ret(ci,i) = self[ci][i];
                        }
                        for(size_t i=self[ci].size(); i<4; ++i){
                            ret(ci,i) = 0;
                        }
                    }
                    return ret;
                }) 
            ;
            exportCellVector<Cells0BoundsVector2D>(m, clsVec);

        }
        // cell 1 bounds
        {
            typedef CellBounds<2,1> Cell1Bounds2D;
            typedef CellBoundsVector<2,1> Cell1BoundsVector2D;

            const std::string clsName = std::string("Cell1Bounds2D");
            auto cls = py::class_<Cell1Bounds2D>(m, clsName.c_str());
            exportCellBoundsT<2, 1, Cell1Bounds2D>(m, cls);

            const std::string clsNameVec = std::string("Cell1BoundsVector2D");
            auto clsVec = py::class_<Cell1BoundsVector2D>(m, clsNameVec.c_str());
            clsVec
                .def("__array__",[](const Cell1BoundsVector2D & self){
                    nifty::marray::PyView<uint32_t> ret({size_t(self.size()),size_t(2)});
                    for(uint32_t ci=0 ;ci<self.size(); ++ci){
                        ret(ci,0) = self[ci][0];
                        ret(ci,1) = self[ci][1];
                    }
                    return ret;
                })
            ;

            exportCellVector<Cell1BoundsVector2D>(m, clsVec);
        }




        // cell 1 bounded by (in 2D junctions of boundaries)
        {
            typedef CellBoundedBy<2,1> Cell1BoundedBy2D;
            typedef CellBoundedByVector<2,1> Cell1BoundedByVector2D;

            const std::string clsName = std::string("Cell1BoundedBy2D");
            auto cls = py::class_<Cell1BoundedBy2D>(m, clsName.c_str());
            exportCellBoundsT<2, 1, Cell1BoundedBy2D>(m, cls);

            const std::string clsNameVec = std::string("Cell1BoundedByVector2D");
            auto clsVec = py::class_<Cell1BoundedByVector2D>(m, clsNameVec.c_str());
            clsVec
                .def(py::init<const CellBoundsVector<2,0 > &>())
                .def("__array__",[](const Cell1BoundedByVector2D & self){
                    nifty::marray::PyView<uint32_t> ret({size_t(self.size()),size_t(2)});
                    for(uint32_t ci=0 ;ci<self.size(); ++ci){
                        const auto & b = self[ci];
                        ret(ci,0) = b[0];
                        ret(ci,1) = b[1];
                    }
                    return ret;
                })
                .def("cellsWithCertainBoundedBySize",[](const Cell1BoundedByVector2D & self,const size_t size){

                    std::vector<uint32_t>  cell1Labels;

                    for(uint32_t ci=0 ;ci<self.size(); ++ci){
                        const auto & b = self[ci];
                        if(b.size() == size){
                            cell1Labels.push_back(ci+1);
                        }
                    }


                    nifty::marray::PyView<uint32_t> ret({size_t(cell1Labels.size())});
                    for(auto i=0; i<ret.size(); ++i){
                        ret[i] = cell1Labels[i];
                    }
                    return ret;
                })
            ;

            exportCellVector<Cell1BoundedByVector2D>(m, clsVec);
        }


        // cell 2 bounded by (in 2D boundaries of regions)
        {
            typedef CellBoundedBy<2,2> Cell2BoundedBy2D;
            typedef CellBoundedByVector<2,2> Cell2BoundedByVector2D;

            const std::string clsName = std::string("Cell2BoundedBy2D");
            auto cls = py::class_<Cell2BoundedBy2D>(m, clsName.c_str());
            exportCellBoundsT<2, 2, Cell2BoundedBy2D>(m, cls);

            const std::string clsNameVec = std::string("Cell2BoundedByVector2D");
            auto clsVec = py::class_<Cell2BoundedByVector2D>(m, clsNameVec.c_str());
            clsVec
                .def(py::init<const CellBoundsVector<2,1 > &>())
                //.def("__array__",[](const Cell2BoundedByVector2D & self){
                //    nifty::marray::PyView<uint32_t> ret({size_t(self.size()),size_t(2)});
                //    for(uint32_t ci=0 ;ci<self.size(); ++ci){
                //        ret(ci,0) = self[ci][0];
                //        ret(ci,1) = self[ci][1];
                //    }
                //    return ret;
                //})
            ;

            exportCellVector<Cell2BoundedByVector2D>(m, clsVec);
        }






        // bounds
        {
            typedef TopologicalGrid<2> TopologicalGridType;
            typedef Bounds<2> BoundsType;

            const std::string clsName = std::string("Bounds2D");
            py::class_<BoundsType>(m, clsName.c_str())

                .def(py::init<const TopologicalGridType &>())
                //.def("__init__",[](
                //    BoundsType & self,
                //    const TopologicalGridType & tgrid
                //){
                //    new (&self) BoundsType(tgrid);
                //})
                //.def("bounds",
                //)
                .def("cell0Bounds",[](
                    const BoundsType & self
                ){
                    //return self.bounds0_;
                    return self. template bounds<0>();
                    //std::cout<<"addr 0 "<<&r<<"\n";
                    //return r;

                },py::return_value_policy::reference_internal)

                .def("cell1Bounds",[](
                    const BoundsType & self
                ){
                    //return self.bounds1_;
                    return self. template bounds<1>();
                    //std::cout<<"addr 1 "<<&r<<"\n";
                    //return r;

                },py::return_value_policy::reference_internal)
            ;
        }


    }




    void exportBounds(py::module & m) {
        exportBounds2D(m);
    }

}
}
