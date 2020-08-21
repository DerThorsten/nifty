
#include <pybind11/pybind11.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "nifty/array/static_array.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/transformation/transformation_functions.hxx"
#include "nifty/transformation/coordinate_transformation_chunked.hxx"

namespace py = pybind11;

namespace nifty{
namespace transformation{

    // in-memory version of the transformation, not implemented yet
    /*
    template<class ARRAY, unsigned NDIM>
    void exportCoordinateTransformationT(py::module & m, const std::string & name) {
        const std::string fuName = "coordinateTransformation" + name;
        m.def(fuName.c_str(), [](const ARRAY & input, const xt::pytensor<double, 2> & matrix,
                                 const array::StaticArray<int64_t, NDIM> & start,
                                 const array::StaticArray<int64_t, NDIM> & stop,
                                 const double fillValue){
            typedef typename ARRAY::shape_type ShapeType;
            ShapeType outShape;
            for(unsigned d = 0; d < NDIM; ++d) {
                outShape[d] = stop[d] - start[d];
            }

            ARRAY out = fillValue * xt::ones<typename ARRAY::value_type>(outShape);
            {
                py::gil_scoped_release allowThreads;
                auto trafo = [&matrix](const array::StaticArray<int64_t, NDIM> & inCoord,
                                       array::StaticArray<double, NDIM> & coord){
                    return affineCoordinateTransformation<NDIM>(inCoord, coord, matrix);
                };
                if(order == 0){
                    coordinateTransformation<NDIM>(input, out, trafo,
                                                   intepolateNearest<NDIM>, start, stop);
                } else if(order == 1){
                    coordinateTransformation<NDIM>(input, out, trafo,
                                                   intepolateLinear<NDIM>, start, stop);
                } else {
                    throw std::invalid_argument("Invalid interpolation order");
                }
            }
            return out;
        });
    }

    void exportCoordinateTransformation(py::module & m) {
        // export 2d coordinate transformations
        // float types
        typedef xt::pytensor<float, 2> Float2DArray;
        exportCoordinateTransformationT<Float2DArray, 2>(m, "2Dfloat32");
        typedef xt::pytensor<double, 2> Double2DArray;
        exportCoordinateTransformationT<Double2DArray, 2>(m, "2Dfloat64");

        // int types
        typedef xt::pytensor<uint8_t, 2> UByte2DArray;
        exportCoordinateTransformationT<UByte2DArray, 2>(m, "2Duint8");
        typedef xt::pytensor<int8_t, 2> Byte2DArray;
        exportCoordinateTransformationT<Byte2DArray, 2>(m, "2Dint8");

        typedef xt::pytensor<uint16_t, 2> UShort2DArray;
        exportCoordinateTransformationT<UShort2DArray, 2>(m, "2Duint16");
        typedef xt::pytensor<int16_t, 2> Short2DArray;
        exportCoordinateTransformationT<Short2DArray, 2>(m, "2Dint16");

        typedef xt::pytensor<uint32_t, 2> UInt2DArray;
        exportCoordinateTransformationT<UInt2DArray, 2>(m, "2Duint32");
        typedef xt::pytensor<int32_t, 2> Int2DArray;
        exportCoordinateTransformationT<Int2DArray, 2>(m, "2Dint32");

        typedef xt::pytensor<uint64_t, 2> ULong2DArray;
        exportCoordinateTransformationT<ULong2DArray, 2>(m, "2Duint64");
        typedef xt::pytensor<int64_t, 2> Long2DArray;
        exportCoordinateTransformationT<Long2DArray, 2>(m, "2Dint64");

        // export 3d coordinate transformations
        // float types
        typedef xt::pytensor<float, 3> Float3DArray;
        exportCoordinateTransformationT<Float3DArray, 3>(m, "3Dfloat32");
        typedef xt::pytensor<double, 3> Double3DArray;
        exportCoordinateTransformationT<Double3DArray, 3>(m, "3Dfloat64");

        // int types
        typedef xt::pytensor<uint8_t, 3> UByte3DArray;
        exportCoordinateTransformationT<UByte3DArray, 3>(m, "3Duint8");
        typedef xt::pytensor<int8_t, 3> Byte3DArray;
        exportCoordinateTransformationT<Byte3DArray, 3>(m, "3Dint8");

        typedef xt::pytensor<uint16_t, 3> UShort3DArray;
        exportCoordinateTransformationT<UShort3DArray, 3>(m, "3Duint16");
        typedef xt::pytensor<int16_t, 3> Short3DArray;
        exportCoordinateTransformationT<Short3DArray, 3>(m, "3Dint16");

        typedef xt::pytensor<uint32_t, 3> UInt3DArray;
        exportCoordinateTransformationT<UInt3DArray, 3>(m, "3Duint32");
        typedef xt::pytensor<int32_t, 3> Int3DArray;
        exportCoordinateTransformationT<Int3DArray, 3>(m, "3Dint32");

        typedef xt::pytensor<uint64_t, 3> ULong3DArray;
        exportCoordinateTransformationT<ULong3DArray, 3>(m, "3Duint64");
        typedef xt::pytensor<int64_t, 3> Long3DArray;
        exportCoordinateTransformationT<Long3DArray, 3>(m, "3Dint64");
    }
    */

    #ifdef WITH_Z5
    template<class T, unsigned NDIM>
    void exportCoordinateTransformationZ5T(py::module & m, const std::string & name) {
        const std::string fuName = "coordinateTransformationZ5" + name;
        m.def(fuName.c_str(), [](const std::string & path, const std::string & key,
                                 const std::string & coordinateFile,
                                 const array::StaticArray<int64_t, NDIM> & start,
                                 const array::StaticArray<int64_t, NDIM> & stop,
                                 const double fillValue){
            typedef xt::pytensor<T, NDIM> ArrayType;
            typedef typename ArrayType::shape_type ShapeType;
            typedef array::StaticArray<int64_t, NDIM> CoordType;

            ShapeType outShape;
            for(unsigned d = 0; d < NDIM; ++d) {
                outShape[d] = stop[d] - start[d];
            }

            ArrayType out = fillValue * xt::ones<T>(outShape);
            nz5::DatasetWrapper<T> input(path, key);
            {
                py::gil_scoped_release allowThreads;

                std::vector<CoordType> inputCoordinates;
                std::vector<CoordType> outputCoordinates;
                parseTransformixCoordinates<NDIM>(coordinateFile, inputCoordinates, outputCoordinates);

                coordinateListTransformationChunked<NDIM>(input, out,
                                                          inputCoordinates, outputCoordinates,
                                                          start);
            }
            return out;
        });
    }


    void exportCoordinateTransformationZ5(py::module & m) {
        // export 2d coordinate transformations
        // float types
        exportCoordinateTransformationZ5T<float, 2>(m, "2Dfloat32");
        exportCoordinateTransformationZ5T<double, 2>(m, "2Dfloat64");

        // int types
        exportCoordinateTransformationZ5T<uint8_t, 2>(m, "2Duint8");
        exportCoordinateTransformationZ5T<int8_t, 2>(m, "2Dint8");
        exportCoordinateTransformationZ5T<uint16_t, 2>(m, "2Duint16");
        exportCoordinateTransformationZ5T<int16_t, 2>(m, "2Dint16");
        exportCoordinateTransformationZ5T<uint32_t, 2>(m, "2Duint32");
        exportCoordinateTransformationZ5T<int32_t, 2>(m, "2Dint32");
        exportCoordinateTransformationZ5T<uint64_t, 2>(m, "2Duint64");
        exportCoordinateTransformationZ5T<int64_t, 2>(m, "2Dint64");

        // export 3d coordinate transformations
        // float types
        exportCoordinateTransformationZ5T<float, 3>(m, "3Dfloat32");
        exportCoordinateTransformationZ5T<double, 3>(m, "3Dfloat64");

        // int types
        exportCoordinateTransformationZ5T<uint8_t, 3>(m, "3Duint8");
        exportCoordinateTransformationZ5T<int8_t, 3>(m, "3Dint8");
        exportCoordinateTransformationZ5T<uint16_t, 3>(m, "3Duint16");
        exportCoordinateTransformationZ5T<int16_t, 3>(m, "3Dint16");
        exportCoordinateTransformationZ5T<uint32_t, 3>(m, "3Duint32");
        exportCoordinateTransformationZ5T<int32_t, 3>(m, "3Dint32");
        exportCoordinateTransformationZ5T<uint64_t, 3>(m, "3Duint64");
        exportCoordinateTransformationZ5T<int64_t, 3>(m, "3Dint64");
    }
    #endif

    /* hdf5 version, not implemented yet
    #ifdef WITH_HDF5
    // TODO pre-smooothing and multi-channel
    template<class T, unsigned NDIM>
    void exportCoordinateTransformationH5T(py::module & m, const std::string & name) {
        const std::string fuName = "coordinateTransformationH5" + name;
        m.def(fuName.c_str(), [](const std::string & path, const std::string & key,
                                 const xt::pytensor<double, 2> & matrix, const int order,
                                 const array::StaticArray<int64_t, NDIM> & start,
                                 const array::StaticArray<int64_t, NDIM> & stop,
                                 const double fillValue){
            typedef xt::pytensor<T, NDIM> ArrayType;
            typedef typename ArrayType::shape_type ShapeType;
            ShapeType outShape;
            for(unsigned d = 0; d < NDIM; ++d) {
                outShape[d] = stop[d] - start[d];
            }

            const auto hidt = hdf5::openFile(path, hdf5::FileAccessMode::READ_ONLY);
            hdf5::Hdf5Array<T> input(hidt, key);
            ArrayType out = fillValue * xt::ones<T>(outShape);
            {
                py::gil_scoped_release allowThreads;
                auto trafo = [&matrix](const array::StaticArray<int64_t, NDIM> & inCoord,
                                       array::StaticArray<double, NDIM> & coord){
                    return coordinateCoordinateTransformation<NDIM>(inCoord, coord, matrix);
                };
                if(order == 0){
                    coordinateTransformationChunked<NDIM>(input, out, trafo,
                                                          intepolateNearest<NDIM>, start, stop);
                } else if(order == 1){
                    coordinateTransformationChunked<NDIM>(input, out, trafo,
                                                          intepolateLinear<NDIM>, start, stop);
                } else {
                    throw std::invalid_argument("Invalid interpolation order");
                }
            }
            hdf5::closeFile(hidt);
            return out;
        });
    }

    void exportCoordinateTransformationH5(py::module & m) {
        // export 2d coordinate transformations
        // float types
        exportCoordinateTransformationH5T<float, 2>(m, "2Dfloat32");
        exportCoordinateTransformationH5T<double, 2>(m, "2Dfloat64");

        // int types
        exportCoordinateTransformationH5T<uint8_t, 2>(m, "2Duint8");
        exportCoordinateTransformationH5T<int8_t, 2>(m, "2Dint8");
        exportCoordinateTransformationH5T<uint16_t, 2>(m, "2Duint16");
        exportCoordinateTransformationH5T<int16_t, 2>(m, "2Dint16");
        exportCoordinateTransformationH5T<uint32_t, 2>(m, "2Duint32");
        exportCoordinateTransformationH5T<int32_t, 2>(m, "2Dint32");
        exportCoordinateTransformationH5T<uint64_t, 2>(m, "2Duint64");
        exportCoordinateTransformationH5T<int64_t, 2>(m, "2Dint64");

        // export 3d coordinate transformations
        // float types
        exportCoordinateTransformationH5T<float, 3>(m, "3Dfloat32");
        exportCoordinateTransformationH5T<double, 3>(m, "3Dfloat64");

        // int types
        exportCoordinateTransformationH5T<uint8_t, 3>(m, "3Duint8");
        exportCoordinateTransformationH5T<int8_t, 3>(m, "3Dint8");
        exportCoordinateTransformationH5T<uint16_t, 3>(m, "3Duint16");
        exportCoordinateTransformationH5T<int16_t, 3>(m, "3Dint16");
        exportCoordinateTransformationH5T<uint32_t, 3>(m, "3Duint32");
        exportCoordinateTransformationH5T<int32_t, 3>(m, "3Dint32");
        exportCoordinateTransformationH5T<uint64_t, 3>(m, "3Duint64");
        exportCoordinateTransformationH5T<int64_t, 3>(m, "3Dint64");
    }
    #endif
    */
}
}
