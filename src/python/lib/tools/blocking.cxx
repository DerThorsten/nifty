#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "nifty/python/converter.hxx"
#include "nifty/tools/blocking.hxx"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;



namespace nifty{
namespace tools{


    template<std::size_t DIM>
    void exportBlockingT(py::module & toolsModule){
        const auto dimStr = std::to_string(DIM) + std::string("d");


        typedef Blocking<DIM, int64_t> BlockingType;
        typedef typename BlockingType::VectorType VectorType;
        typedef typename BlockingType::BlockType BlockType;
        typedef typename BlockingType::BlockWithHaloType BlockWithHaloType;



        const auto blockClsStr = std::string("Block") + dimStr;
        py::class_<BlockType>(toolsModule, blockClsStr.c_str())

            .def_property_readonly("begin",&BlockType::begin)
            .def_property_readonly("end",&BlockType::end)
            .def_property_readonly("shape",&BlockType::shape)

			.def(py::pickle(
                 [](const BlockType &b) { // __getstate__
                     /* Return a tuple that fully encodes the state of the object */
                     return py::make_tuple(b.begin().asStdArray(),
                                           b.end().asStdArray());
                 },
                 [](py::tuple t) { // __setstate__
                     if (t.size() != 2)
                         throw std::runtime_error("Invalid state!");

                     const auto & pBeg = t[0].cast<typename VectorType::BaseType>();
                     const auto & pEnd = t[1].cast<typename VectorType::BaseType>();

                     VectorType beg, end;
                     for(unsigned d = 0; d < DIM; ++d) {
                        beg[d] = pBeg[d];
                        end[d] = pEnd[d];
                     }
                     BlockType b(beg, end);
                     return b;
                 }
            ))
        ;

        const auto blockWithHaloClsStr = std::string("BlockWithHalo") + dimStr;
        py::class_<BlockWithHaloType>(toolsModule, blockWithHaloClsStr.c_str())

            .def_property_readonly("outerBlock",&BlockWithHaloType::outerBlock ,py::return_value_policy::reference_internal)
            .def_property_readonly("innerBlock",&BlockWithHaloType::innerBlock ,py::return_value_policy::reference_internal)
            .def_property_readonly("innerBlockLocal",&BlockWithHaloType::innerBlockLocal ,py::return_value_policy::reference_internal)

            .def(py::pickle(
                 [](const BlockWithHaloType &b) { // __getstate__
                     /* Return a tuple that fully encodes the state of the object */
                     return py::make_tuple(b.outerBlock(), b.innerBlock());
                 },
                 [](py::tuple t) { // __setstate__
                     if (t.size() != 2)
                         throw std::runtime_error("Invalid state!");

                     const auto & outer = t[0].cast<BlockType>();
                     const auto & inner = t[1].cast<BlockType>();
                     BlockWithHaloType b(outer, inner);
                     return b;
                 }
            ))
        ;

        const auto blockingClsStr = std::string("Blocking") + dimStr;
        py::class_<BlockingType>(toolsModule, blockingClsStr.c_str())

            .def(py::init([](VectorType roiBegin,
                             VectorType roiEnd,
                             VectorType blockShape,
                             VectorType blockShift){
                    return new BlockingType(roiBegin, roiEnd, blockShape, blockShift);
                })
            )

            .def_property_readonly("roiBegin",&BlockingType::roiBegin)
            .def_property_readonly("roiEnd",&BlockingType::roiEnd)
            .def_property_readonly("blockShape",&BlockingType::blockShape)
            .def_property_readonly("blockShift",&BlockingType::blockShift)
            .def_property_readonly("blocksPerAxis",&BlockingType::blocksPerAxis)
            .def_property_readonly("numberOfBlocks",&BlockingType::numberOfBlocks)

            .def(py::pickle(
                 [](const BlockingType &b) { // __getstate__
                     /* Return a tuple that fully encodes the state of the object */
                     return py::make_tuple(b.roiBegin().asStdArray(),
                                           b.roiEnd().asStdArray(),
                                           b.blockShape().asStdArray(),
                                           b.blockShift().asStdArray());
                 },
                 [](py::tuple t) { // __setstate__
                     if (t.size() != 4)
                         throw std::runtime_error("Invalid state!");

                     const auto & pBeg = t[0].cast<typename VectorType::BaseType>();
                     const auto & pEnd = t[1].cast<typename VectorType::BaseType>();
                     const auto & pShape = t[2].cast<typename VectorType::BaseType>();
                     const auto & pShift = t[3].cast<typename VectorType::BaseType>();

                     VectorType beg, end, shape, shift;
                     for(unsigned d = 0; d < DIM; ++d) {
                        beg[d] = pBeg[d];
                        end[d] = pEnd[d];
                        shape[d] = pShape[d];
                        shift[d] = pShift[d];
                     }
                     BlockingType b(beg, end, shape, shift);
                     return b;
                 }
            ))

            .def("getBlock", &BlockingType::getBlock)

            .def("getBlockIdsInBoundingBox", [](const BlockingType & self,
                const VectorType roiBegin,
                const VectorType roiEnd) {

                std::vector<uint64_t> tmp;
                {
                    py::gil_scoped_release allowThreads;
                    self.getBlockIdsInBoundingBox(roiBegin, roiEnd, tmp);
                }
                xt::pytensor<uint64_t, 1> out = xt::zeros<uint64_t>({tmp.size()});
                {
                    py::gil_scoped_release allowThreads;
                    for(int i = 0; i < tmp.size(); ++i) {
                        out(i) = tmp[i];
                    }
                }
                return out;
            }, py::arg("roiBegin"), py::arg("roiEnd"))

            .def("getBlockIdsOverlappingBoundingBox", [](const BlockingType & self,
                                                         const VectorType & roiBegin,
                                                         const VectorType & roiEnd) {
                std::vector<uint64_t> tmp;
                {
                    py::gil_scoped_release allowThreads;
                    self.getBlockIdsOverlappingBoundingBox(roiBegin, roiEnd, tmp);
                }
                xt::xtensor<uint64_t, 1> out = xt::zeros<uint64_t>({tmp.size()});
                {
                    py::gil_scoped_release allowThreads;
                    for(int i = 0; i < tmp.size(); ++i) {
                        out(i) = tmp[i];
                    }
                }
                return out;
            }, py::arg("roiBegin"), py::arg("roiEnd"))

            .def("getLocalOverlaps", [](
                const BlockingType & self,
                const std::size_t indexA,
                const std::size_t indexB,
                const VectorType & blockHalo) {

                VectorType blockABegin, blockBBegin, blockAEnd, blockBEnd;
                bool ret;
                {
                    py::gil_scoped_release allowThreads;
                    ret = self.getLocalOverlaps(indexA, indexB, blockHalo,
                                                blockABegin, blockAEnd,
                                                blockBBegin, blockBEnd);
                }
                return std::make_tuple(ret, blockABegin, blockAEnd, blockBBegin, blockBEnd);
            })

            .def("blockGridPosition", [](
                    const BlockingType & self,
                    const uint64_t blockIndex) {
                VectorType gridPosition;
                {
                    py::gil_scoped_release allowThreads;
                    self.blockGridPosition(blockIndex, gridPosition);
                }
                return gridPosition;
            }, py::arg("blockIndex"))

            .def("getBlockWithHalo", [](
                    const BlockingType & self,
                    const std::size_t blockIndex,
                    const VectorType & halo
                ){
                    return self.getBlockWithHalo(blockIndex, halo);
                },
                py::arg("blockIndex"),py::arg("halo")
            )

            .def("getBlockWithHalo", [](
                    const BlockingType & self,
                    const std::size_t blockIndex,
                    const VectorType & haloBegin,
                    const VectorType & haloEnd
                ){
                    return self.getBlockWithHalo(blockIndex, haloBegin, haloEnd);
                },
                py::arg("blockIndex"),py::arg("haloBegin"),py::arg("haloEnd")
            )

            .def("addHalo", [](
                    const BlockingType & self,
                    const BlockType & block,
                    const VectorType halo
                ){
                    return self.addHalo(block, halo);
                },
                py::arg("block"),py::arg("halo")
            )
            .def("addHalo", [](
                    const BlockingType & self,
                    const BlockType & block,
                    const VectorType & haloBegin,
                    const VectorType & haloEnd
                ){
                    return self.addHalo(block, haloBegin, haloEnd);
                },
                py::arg("block"),py::arg("haloBegin"),py::arg("haloEnd")
            )
            .def("getNeighborId", [](
                    const BlockingType & self,
                    const uint64_t blockId,
                    const unsigned axis,
                    const bool lower
            ){
                return self.getNeighborId(blockId, axis, lower);
             },
             py::arg("blockId"), py::arg("axis"), py::arg("lower")
            )
            .def("coordinatesToBlockId", [](const BlockingType & self,
                                            const VectorType & coordinates){
                return self.coordinatesToBlockId(coordinates);
            }, py::arg("coordinates"))
        ;
    }


    void exportBlocking(py::module & toolsModule) {
        exportBlockingT<1>(toolsModule);
        exportBlockingT<2>(toolsModule);
        exportBlockingT<3>(toolsModule);
        exportBlockingT<4>(toolsModule);
    }

}
}
