#pragma once
#include <nifty/xtensor/xtensor.hxx>


namespace nifty {
namespace filters {


    template<class POINTS>
    inline void computePointDistances(const POINTS & points, xt::xtensor<float, 2> & pointDistances) {
        const std::size_t nPoints = points.shape()[0];
        const unsigned ndim = points.shape()[1];

        for(std::size_t i = 0; i < nPoints; ++i) {
            for(std::size_t j = i + 1; j < nPoints; ++j) {

                float dist = 0;
                for(unsigned d = 0; d < ndim; ++d) {
                    const float diff = (static_cast<float>(points(i, d)) - static_cast<float>(points(j, d)));
                    dist += diff * diff;
                }
                dist = sqrt(dist);

                pointDistances(i, j) = dist;
                pointDistances(j, i) = dist;
            }
        }
    }


    template<class DISTANCE_MAP, class POINTS>
    inline uint64_t findBestPoint(const uint64_t pointId, const float thisDistance, const DISTANCE_MAP & distanceMap,
                                  const POINTS & points, const xt::xtensor<float, 2> & pointDistances) {
        const std::size_t nPoints = pointDistances.shape()[0];
        const unsigned ndim = points.shape()[1];
        float maxDistance = -std::numeric_limits<float>::max();
        uint64_t bestPoint = pointId;

        xt::xindex pointCoord(ndim);
        for(std::size_t i = 0; i < nPoints; ++i) {
            if(pointDistances(pointId, i) > thisDistance) {
                continue;
            }

            for(unsigned d = 0; d < ndim; ++d) {
                pointCoord[d] = points(i, d);
            }
            const float dist = distanceMap[pointCoord];
            if(dist > maxDistance) {
                bestPoint = i;
                maxDistance = dist;
            }
        }

        return bestPoint;
    }


    template<class DISTANCE_MAP, class POINTS>
    inline void nonMaximumDistanceSuppression(const DISTANCE_MAP & distanceMap, const POINTS & points,
                                              std::set<uint64_t> & pointsOut) {
        const int64_t nPoints = points.shape()[0];
        const unsigned ndim = points.shape()[1];

        // compute the euclidean distances between all points
        xt::xtensor<float, 2> pointDistances = xt::zeros<float>({nPoints, nPoints});
        computePointDistances(points, pointDistances);

        xt::xindex pointCoord(ndim);
        // iterate over all points and find the closest seed
        for(std::size_t pointId = 0; pointId < nPoints; ++pointId) {
            for(unsigned d = 0; d < ndim; ++d) {
                pointCoord[d] = points(pointId, d);
            }
            const float dist = distanceMap[pointCoord];
            pointsOut.insert(findBestPoint(pointId, dist, distanceMap, points, pointDistances));
        }
    }

}
}
