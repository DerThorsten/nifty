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
            typedef std::vector<Cell0Bounds2D> Cells0BoundsVector2D;

            const std::string clsName = std::string("Cells0Bounds2D");
            auto cls = py::class_<Cell0Bounds2D>(m, clsName.c_str());
            exportCellBoundsT<2, 0, Cell0Bounds2D>(m, cls);

            const std::string clsNameVec = std::string("Cells0BoundsVector2D");
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
            typedef std::vector<Cell1Bounds2D> Cell1BoundsVector2D;

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
