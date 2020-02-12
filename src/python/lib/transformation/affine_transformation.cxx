#include <pybind11/pybind11.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "nifty/array/static_array.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/transformation/affine_transformation.hxx"
#include "nifty/transformation/coordinate_transformation.hxx"
#include "nifty/transformation/coordinate_transformation_chunked.hxx"

namespace py = pybind11;

namespace nifty{
namespace transformation{

    // TODO multi-channel
    template<class ARRAY, unsigned NDIM>
    void exportAffineTransformationT(py::module & m, const std::string & name) {
        const std::string fuName = "affineTransformation" + name;
        m.def(fuName.c_str(), [](const ARRAY & input, const xt::pytensor<double, 2> & matrix,
                                 const int order,
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


    void exportAffineTransformation(py::module & m) {
        // export 2d affine transformations
        // float types
        typedef xt::pytensor<float, 2> Float2DArray;
        exportAffineTransformationT<Float2DArray, 2>(m, "2Dfloat32");
        typedef xt::pytensor<double, 2> Double2DArray;
        exportAffineTransformationT<Double2DArray, 2>(m, "2Dfloat64");

        // int types
        typedef xt::pytensor<uint8_t, 2> UByte2DArray;
        exportAffineTransformationT<UByte2DArray, 2>(m, "2Duint8");
        typedef xt::pytensor<int8_t, 2> Byte2DArray;
        exportAffineTransformationT<Byte2DArray, 2>(m, "2Dint8");

        typedef xt::pytensor<uint16_t, 2> UShort2DArray;
        exportAffineTransformationT<UShort2DArray, 2>(m, "2Duint16");
        typedef xt::pytensor<int16_t, 2> Short2DArray;
        exportAffineTransformationT<Short2DArray, 2>(m, "2Dint16");

        // export 3d affine transformations
        // float types
        typedef xt::pytensor<float, 3> Float3DArray;
        exportAffineTransformationT<Float3DArray, 3>(m, "3Dfloat32");
        typedef xt::pytensor<double, 3> Double3DArray;
        exportAffineTransformationT<Double3DArray, 3>(m, "3Dfloat64");

        // int types
        typedef xt::pytensor<uint8_t, 3> UByte3DArray;
        exportAffineTransformationT<UByte3DArray, 3>(m, "3Duint8");
        typedef xt::pytensor<int8_t, 3> Byte3DArray;
        exportAffineTransformationT<Byte3DArray, 3>(m, "3Dint8");

        typedef xt::pytensor<uint16_t, 3> UShort3DArray;
        exportAffineTransformationT<UShort3DArray, 3>(m, "3Duint16");
        typedef xt::pytensor<int16_t, 3> Short3DArray;
        exportAffineTransformationT<Short3DArray, 3>(m, "3Dint16");
    }


    #ifdef WITH_Z5
    // TODO pre-smooothing and multi-channel
    template<class T, unsigned NDIM>
    void exportAffineTransformationZ5T(py::module & m, const std::string & name) {
        const std::string fuName = "affineTransformationZ5" + name;
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

            ArrayType out = fillValue * xt::ones<T>(outShape);
            nz5::DatasetWrapper<T> input(path, key);
            {
                py::gil_scoped_release allowThreads;
                auto trafo = [&matrix](const array::StaticArray<int64_t, NDIM> & inCoord,
                                       array::StaticArray<double, NDIM> & coord){
                    return affineCoordinateTransformation<NDIM>(inCoord, coord, matrix);
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
            return out;
        });
    }


    void exportAffineTransformationZ5(py::module & m) {
        // export 2d affine transformations
        // float types
        exportAffineTransformationZ5T<float, 2>(m, "2Dfloat32");
        exportAffineTransformationZ5T<double, 2>(m, "2Dfloat64");

        // int types
        exportAffineTransformationZ5T<uint8_t, 2>(m, "2Duint8");
        exportAffineTransformationZ5T<int8_t, 2>(m, "2Dint8");
        exportAffineTransformationZ5T<uint16_t, 2>(m, "2Duint16");
        exportAffineTransformationZ5T<int16_t, 2>(m, "2Dint16");

        // export 3d affine transformations
        // float types
        exportAffineTransformationZ5T<float, 3>(m, "3Dfloat32");
        exportAffineTransformationZ5T<double, 3>(m, "3Dfloat64");

        // int types
        exportAffineTransformationZ5T<uint8_t, 3>(m, "3Duint8");
        exportAffineTransformationZ5T<int8_t, 3>(m, "3Dint8");
        exportAffineTransformationZ5T<uint16_t, 3>(m, "3Duint16");
        exportAffineTransformationZ5T<int16_t, 3>(m, "3Dint16");
    }
    #endif


    #ifdef WITH_HDF5
    // TODO pre-smooothing and multi-channel
    template<class T, unsigned NDIM>
    void exportAffineTransformationH5T(py::module & m, const std::string & name) {
        const std::string fuName = "affineTransformationH5" + name;
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
                    return affineCoordinateTransformation<NDIM>(inCoord, coord, matrix);
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


    void exportAffineTransformationH5(py::module & m) {
        // export 2d affine transformations
        // float types
        exportAffineTransformationH5T<float, 2>(m, "2Dfloat32");
        exportAffineTransformationH5T<double, 2>(m, "2Dfloat64");

        // int types
        exportAffineTransformationH5T<uint8_t, 2>(m, "2Duint8");
        exportAffineTransformationH5T<int8_t, 2>(m, "2Dint8");
        exportAffineTransformationH5T<uint16_t, 2>(m, "2Duint16");
        exportAffineTransformationH5T<int16_t, 2>(m, "2Dint16");

        // export 3d affine transformations
        // float types
        exportAffineTransformationH5T<float, 3>(m, "3Dfloat32");
        exportAffineTransformationH5T<double, 3>(m, "3Dfloat64");

        // int types
        exportAffineTransformationH5T<uint8_t, 3>(m, "3Duint8");
        exportAffineTransformationH5T<int8_t, 3>(m, "3Dint8");
        exportAffineTransformationH5T<uint16_t, 3>(m, "3Duint16");
        exportAffineTransformationH5T<int16_t, 3>(m, "3Dint16");
    }
    #endif
}
}
