#pragma once
#include <fstream>
#include "nifty/array/static_array.hxx"
#include "nifty/xtensor/xtensor.hxx"


namespace nifty {
namespace transformation {

    template<unsigned NDIM, class MATRIX>
    inline void affineCoordinateTransformation(const array::StaticArray<int64_t, NDIM> & inCoord,
                                               array::StaticArray<double, NDIM> & coord,
                                               const MATRIX & matrix) {
        for(unsigned di = 0; di < NDIM; ++di) {
            coord[di] = 0.;
            for(unsigned dj = 0; dj < NDIM; ++dj) {
                coord[di] += matrix(di, dj) * inCoord[dj];
            }
            coord[di] += matrix(di, NDIM);
        }
    }

    template<unsigned NDIM, class COORD_TYPE>
    inline void parseTransformixCoordinates(const std::string & coordPath,
                                            std::vector<COORD_TYPE> & inputCoordinates,
                                            std::vector<COORD_TYPE> & outputCoordinates) {

        typedef COORD_TYPE CoordType;
        const std::string searchString1 = "InputIndex = [ ";
        const std::string searchString2 = "OutputIndexFixed = [ ";

        std::ifstream f(coordPath);
        std::string line;

        CoordType inCoord, outCoord;

        // transformix maps coordinates from output space to input space, so what is
        // called "InputIndex" in the transformix file corresponds to our output coordinate
        // and what is called "OutputIndexFixed" corresponds to our input coordinate
        // also, transformix uses the reversed axis convention (e.g. XYZ instead of ZYX)
        while(std::getline(f, line)) {
            // read the output coordinate
            auto p0 = line.find(searchString1) + searchString1.size();
            auto p1 = line.find(" ", p0);
            for(unsigned d = 0; d < NDIM; ++d) {
                outCoord[NDIM - d - 1] = std::stol(line.substr(p0, p1));
                p0 = p1 + 1;
                p1 = line.find(" ", p0);
            }
            outputCoordinates.push_back(outCoord);

            // read the input coordinate
            p0 = line.find(searchString2) + searchString2.size();
            p1 = line.find(" ", p0);
            for(unsigned d = 0; d < NDIM; ++d) {
                inCoord[NDIM - d - 1] = std::stol(line.substr(p0, p1));
                p0 = p1 + 1;
                p1 = line.find(" ", p0);
            }
            inputCoordinates.push_back(inCoord);
        }
    }

}
}
