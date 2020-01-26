#include <pybind11/pybind11.h>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "nifty/array/static_array.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/transformation/affine_transformation.hxx"
#include "nifty/transformation/coordinate_transformation.hxx"

namespace py = pybind11;

namespace nifty{
namespace transformation{

    // TODO pre-smoothing and multi-channel
    template<class ARRAY, unsigned NDIM>
    void exportAffineTransformationT(py::module & m, const std::string & name) {
        const std::string fuName = "affineTransformation" + name;
        m.def(fuName.c_str(), [](const ARRAY & input, const xt::pytensor<float, 2> & matrix,
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
                                       array::StaticArray<float, NDIM> & coord){
                    return affineCoordinateTransformation<NDIM>(inCoord, coord, matrix);
                };
                if(order == 0){
                    coordinateTransformation<NDIM>(input, out, trafo,
                                                   intepolateNearest<NDIM>, start, stop);
                } else if(order == 1){
                    coordinateTransformation<NDIM>(input, out, trafo,
                                                   intepolateLinear<NDIM>, start, stop);
                } else
                    throw std::invalid_argument("Invalid interpolation order");
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
        typedef xt::pytensor<uint8_t, 2> Byte2DArray;
        exportAffineTransformationT<Byte2DArray, 2>(m, "2Duint8");

        // export 3d affine transformations
        // float types
        typedef xt::pytensor<float, 3> Float3DArray;
        exportAffineTransformationT<Float3DArray, 3>(m, "3Dfloat32");
        typedef xt::pytensor<double, 3> Double3DArray;
        exportAffineTransformationT<Double3DArray, 3>(m, "3Dfloat64");
        // int types
        typedef xt::pytensor<uint8_t, 3> Byte3DArray;
        exportAffineTransformationT<Byte3DArray, 3>(m, "3Duint8");
    }
}
}
