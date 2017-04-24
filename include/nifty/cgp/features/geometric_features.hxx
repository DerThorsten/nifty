#pragma once

#include "nifty/cgp/geometry.hxx"
#include "nifty/marray/marray.hxx"


namespace nifty{
namespace cgp{

    class CurvatureFeatrues{

    public:
        CurvatureFeatrues(){

        }

        size_t numberOfFeatrues()const{
            return 3;    
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,1> & cellsGeometry,
            nifty::marray::View<T> & features
        ){  
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
        templatec<class COORDS>
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
    };
}
}