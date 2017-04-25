#pragma once

#include <vector>

#include "nifty/histogram/histogram.hxx"
#include "nifty/cgp/geometry.hxx"
#include "nifty/marray/marray.hxx"



namespace nifty{
namespace cgp{

    class Cell1CurvatureFeatures2D{

    public:
        Cell1CurvatureFeatures2D(
            const std::vector<float> & sigmas
        )
        :   sigmas_(sigmas)
        {

        }

        size_t numberOfFeatures()const{
            return 3;    
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,1> & cellsGeometry,
            nifty::marray::View<T> & features
        )const{  
            for(auto cellIndex=0; cellIndex<cellsGeometry.size(); ++cellIndex){

                const auto & cellGeometry = cellsGeometry[cellIndex];

                // compute the derivatives
                for(auto i=0; i<cellGeometry.size(); ++i){

                    double dx,dy,dxx,dyy;
                    calcDerivatives(cellGeometry,i,dx,dy,dxx,dyy);

                    // the curvature kappa
                    const double kappa =  std::abs(dx*dyy - dy*dxx)/
                        std::pow(dx*dx+dy*dy,2.0/3.0);

                }

            }
        }   

    private:
        template<class COORDS>
        void calcDerivatives(
            const COORDS & coords,
            const int pos,
            double & dx,
            double & dy,
            double & dxx,
            double & dyy
        )const{

            const double sd[5] = {
                 1.0/12.0,
                -8.0/12.0,
                 0.0/12.0,
                 8.0/12.0,
                -1.0/12.0
            };
            const double sdd[5] = {
                -1.0/12.0,
                 16.0/12.0,
                -30.0/12.0,
                 16.0/12.0,
                -1.0/12.0,
            };

            const int  n = coords.size();

            dx = 0;
            dy = 0;
            dxx = 0;
            dyy = 0;

    

            for(int j=-2; j<=2; ++j){
                // kernel index
                const int ki = j+2;
                // data index
                const int di = std::max(int(0),std::min(n-1, pos+j));


                auto valKd = sd[ki];
                auto valKdd = sdd[ki];

                auto valX = double(coords[di][0]);
                auto valY = double(coords[di][1]);

                dx += valKd*valX;
                dy += valKd*valY;

                dxx += valKdd*valX;
                dyy += valKdd*valY;

            }
            
        }


        std::vector<float> sigmas_;
    };




    class Cell1GeoHistFeatures{

    public:
        Cell1GeoHistFeatures(
            const std::vector<float> & quantiles,
            const size_t bincount = 40,
            const float gamma = -0.07
        )
        :   quantiles_(quantiles),
            bincount_(bincount),
            gamma_(gamma)
        {

        }

        size_t numberOfFeatures()const{
            return quantiles_.size();    
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,1> & cellsGeometry,
            nifty::marray::View<T> & features
        )const{  
            for(auto cellIndex=0; cellIndex<cellsGeometry.size(); ++cellIndex){



                const auto & cellGeometry = cellsGeometry[cellIndex];
                const auto centerOfMass = cellGeometry.centerOfMass();

                const float minVal = 0.0;
                const float maxVal = std::exp(gamma_*1.0);

                histogram::Histogram<float> histogram(minVal, maxVal, bincount_);
                std::vector<float> out(quantiles_.size());

                // compute the derivatives
                for(auto i=0; i<cellGeometry.size(); ++i){

                    const auto & coord = cellGeometry[i];
                    
                    // distance ...(TODO..we need code for that...)
                    auto d = 0.0;
                    for(auto d=0; d<2;++d){
                        auto diff = centerOfMass[d] - coord[d];
                        d += diff * diff;
                    }
                    d = std::sqrt(d);

                    // push to histogram
                    histogram.insert(std::exp(gamma_*d));
                }
                // extract the quantiles 
                quantiles(histogram, quantiles_.begin(), quantiles_.end(), out.begin());

                // write results
                for(auto i=0; i<out.size(); ++i){
                    features(cellIndex, i) = out[i];
                }

            }
        }



    private:

    


        std::vector<float> quantiles_;
        size_t bincount_;
        float gamma_;
    };


}
}