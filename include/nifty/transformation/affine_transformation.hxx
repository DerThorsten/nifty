#pragma once
#include "nifty/array/static_array.hxx"
#include "nifty/xtensor/xtensor.hxx"


namespace nifty {
namespace transformation {

    template<unsigned NDIM, class MATRIX>
    inline void affineCoordinateTransformation(const array::StaticArray<int64_t, NDIM> & inCoord,
                                               array::StaticArray<float, NDIM> & coord,
                                               const MATRIX & matrix) {
        for(unsigned di = 0; di < NDIM; ++di) {
            coord[di] = 0.;
            for(unsigned dj = 0; dj < NDIM; ++dj) {
                coord[di] += matrix(di, dj) * inCoord[dj] + matrix(di, NDIM);
            }
        }
    }

}
}
