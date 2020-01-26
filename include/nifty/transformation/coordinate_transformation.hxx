#pragma once
#include <xtensor/xarray.hpp>
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/xtensor/xtensor.hxx"


namespace nifty {
namespace transformation {

    //
    // interpolation functions
    //

    template<unsigned NDIM>
    inline void intepolateNearest(const array::StaticArray<float, NDIM> & coord,
                                  std::vector<array::StaticArray<int64_t, NDIM>> & coordList,
                                  std::vector<double> & weightList) {
        coordList.resize(1);
        weightList.resize(1);
        weightList[0] = 1.;
        auto & coordOut = coordList[0];
        for(unsigned d = 0; d < NDIM; ++d) {
            coordOut[d] = static_cast<int64_t>(round(coord[d]));
        }
    }

    template<unsigned NDIM>
    inline void intepolateLinear(const array::StaticArray<float, NDIM> & coord,
                                 std::vector<array::StaticArray<int64_t, NDIM>> & coordList,
                                 std::vector<double> & weightList) {
        const std::size_t nOut = pow(2, NDIM);
        coordList.resize(nOut);
        weightList.resize(nOut);
        // TODO
    }

    //
    // coordinate transformation functions
    //

    template<unsigned NDIM, class ARRAY,
             class COORD_TRAFO, class INTERPOLATOR>
    void coordinateTransformation(const ARRAY & input, ARRAY & output,
                                  COORD_TRAFO && trafo, INTERPOLATOR && interpolator,
                                  const array::StaticArray<int64_t, NDIM> & start,
                                  const array::StaticArray<int64_t, NDIM> & stop){
        typedef array::StaticArray<int64_t, NDIM> CoordType;
        typedef array::StaticArray<float, NDIM> FloatCoordType;

        const auto & shape = input.shape();
        array::StaticArray<int64_t, NDIM> maxRange;
        for(unsigned d = 0; d < NDIM; ++d) {
            maxRange[d] = shape[d] - 1;
        }

        CoordType normalizedOutCoord;
        FloatCoordType coord;
        std::vector<CoordType> coordList;
        std::vector<double> weightList;

        tools::forEachCoordinate(start, stop, [&](const CoordType & outCoord){
            // transform the coordinate
            trafo(outCoord, coord);

            // range check
            for(unsigned d = 0; d < NDIM; ++d){
                if(coord[d] >= maxRange[d] || coord[d] < 0) {
                    return;
                }
            }

            // interpolate the coordinate
            interpolator(coord, coordList, weightList);

            // iterate over the interpolated coords and compute the output value
            double val = 0.;
            for(unsigned i = 0; i < coordList.size(); ++i) {
                val += weightList[i] * xtensor::read(input, coordList[i]);
            }

            for(unsigned d = 0; d < NDIM; ++d){
                normalizedOutCoord[d] = outCoord[d] - start[d];
            }
            xtensor::write(output, normalizedOutCoord, val);
        });
    }

}
}
