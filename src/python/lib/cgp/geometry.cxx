#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "xtensor-python/pytensor.hpp"

#include "export_cell_vector.hxx"

#include "nifty/python/converter.hxx"
#include "nifty/cgp/geometry.hxx"


namespace py = pybind11;

namespace nifty{
namespace cgp{

    template<
        size_t DIM,
        size_t CELL_TYPE,
        class CLS
    >
    void exportCellGeometryT(py::module & m, py::class_<CLS> & pyCls) {
        pyCls
            //.def("__len__", &CLS::size)
            .def("__len__", [](const CLS & self){
                return self.size();
            })
            .def("__getitem__", [](const CLS & self, uint32_t i){
                return self[i];
            })
            .def("centerOfMass",&CLS::centerOfMass)
            .def("__array__", [](const CLS & self){
                xt::pytensor<uint32_t, 2> out({self.size(), DIM});
                for(size_t i=0; i<self.size(); ++i){
                    const auto & coord = self[i];
                    for(size_t d=0; d<DIM; ++d){
                        out(i, d) = coord[d];
                    }
                }
                return out;
            })
        ;
    }

    template<
    class CLS
    >
    inline  void exportCellGeometryVector(pybind11::module & m, pybind11::class_<CLS> & pyCls) {

        exportCellVector<CLS>(m, pyCls);

        pyCls
            .def("centersOfMass", [](const CLS & self){
                xt::pytensor<float, 2> cArray({self.size(),
                                               CLS::value_type::DimensionType::value});
                for(auto i=0; i<self.size(); ++i){
                    const auto com  = self[i].centerOfMass();
                    for(auto d=0; d<CLS::value_type::DimensionType::value; ++d){
                        cArray(i, d) = com[d];
                    }
                }
                return cArray;
            })
        ;
    }


    void exportGeometry2D(py::module & m) {

        
        

        // cell 0 geometry
        {   
            typedef CellGeometry<2,0> Cell0Geometry2D;
            typedef CellGeometryVector<2,0> Cells0GeometryVector2D;

            const std::string clsName = std::string("Cells0Geometry2D");
            auto cls = py::class_<Cell0Geometry2D>(m, clsName.c_str());
            exportCellGeometryT<2, 0, Cell0Geometry2D>(m, cls);

            const std::string clsNameVec = std::string("Cells0GeometryVector2D");
            auto clsVec = py::class_<Cells0GeometryVector2D>(m, clsNameVec.c_str());
            exportCellGeometryVector<Cells0GeometryVector2D>(m, clsVec);

        }
        // cell 1 geometry
        {
            typedef CellGeometry<2,1> Cell1Geometry2D;
            typedef CellGeometryVector<2,1> Cell1GeometryVector2D;

            const std::string clsName = std::string("Cell1Geometry2D");
            auto cls = py::class_<Cell1Geometry2D>(m, clsName.c_str());
            exportCellGeometryT<2, 1, Cell1Geometry2D>(m, cls);

            const std::string clsNameVec = std::string("Cell1GeometryVector2D");
            auto clsVec = py::class_<Cell1GeometryVector2D>(m, clsNameVec.c_str());
            exportCellGeometryVector<Cell1GeometryVector2D>(m, clsVec);
        }
        // cell 2 geometry
        {
            typedef CellGeometry<2,2> Cell1Geometry2D;
            typedef CellGeometryVector<2,2> Cell1GeometryVector2D;

            const std::string clsName = std::string("Cell2Geometry2D");
            auto cls = py::class_<Cell1Geometry2D>(m, clsName.c_str());
            exportCellGeometryT<2, 2, Cell1Geometry2D>(m, cls);

            const std::string clsNameVec = std::string("Cell2GeometryVector2D");
            auto clsVec = py::class_<Cell1GeometryVector2D>(m, clsNameVec.c_str());
            exportCellGeometryVector<Cell1GeometryVector2D>(m, clsVec);
        }
        // geometry
        {
            typedef TopologicalGrid<2> TopologicalGridType;
            typedef Geometry<2> GeometryType;

            const std::string clsName = std::string("Geometry2D");
            py::class_<GeometryType>(m, clsName.c_str())

                .def(py::init<const TopologicalGridType &, const bool, const bool>(),
                    py::arg("topologicalGrid"),
                    py::arg("fill"),
                    py::arg("sort1Cells")

                )

                .def("cell0Geometry",[](const GeometryType & self){
                    return self. template geometry<0>();
                },py::return_value_policy::reference_internal)

                .def("cell1Geometry",[](const GeometryType & self){
                    return self. template geometry<1>();
                },py::return_value_policy::reference_internal)

                .def("cell2Geometry",[](const GeometryType & self){
                    return self. template geometry<2>();
                },py::return_value_policy::reference_internal)
            ;
        }

    
    }




    void exportGeometry(py::module & m) {
        exportGeometry2D(m);
    }

}
}
