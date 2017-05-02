#pragma once

#include <vector>

#include "nifty/math/math.hxx"
#include "nifty/histogram/histogram.hxx"
#include "nifty/cgp/geometry.hxx"
#include "nifty/cgp/bounds.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/filters/gaussian_curvature.hxx"


#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>


namespace nifty{
namespace cgp{

    class Cell1CurvatureFeatures2D{
    public:
        Cell1CurvatureFeatures2D(
            const std::vector<float> & sigmas  = std::vector<float>({1.0f, 2.0f, 4.0f}),
            const std::vector<float> & quantiles = std::vector<float>({0.1f, 0.25f, 0.50f, 0.75f, 0.9f})
        )
        :   sigmas_(sigmas),
            quantiles_(quantiles)
        {

        }

        size_t numberOfFeatures()const{
            return sigmas_.size() * quantiles_.size();    
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,1>  & cell1GeometryVector,
            const CellBoundedByVector<2,1> & cell1BoundedByVector,
            nifty::marray::View<T> & features
        )const{  

            std::vector<float> curvature;
            std::vector<float> quantilesOut(quantiles_.size());
            nifty::histogram::Histogram<float> histogram(0,1.0, 10);

            
            for(auto sigmaIndex=0; sigmaIndex<sigmas_.size(); ++sigmaIndex){
                const auto sigma = sigmas_[sigmaIndex];
                nifty::filters::GaussianCurvature2D<> op(sigma, -1, 2.5);


                for(auto cell1Index=0; cell1Index<cell1GeometryVector.size(); ++cell1Index){
                    const auto & geo = cell1GeometryVector[cell1Index];

                    if(geo.size()>=3){
                        // we use a larger size?
                        if(curvature.capacity() < geo.size()){
                            curvature.resize(geo.size()*2);
                        }
                        else{
                            curvature.resize(geo.size());
                        }

                        // calculate curvature
                        //std::cout<<"    is closed "<<"\n";
                        const auto closedLine = cell1BoundedByVector[cell1Index].size() == 0;
                        //std::cout<<"    calculate curvature "<<"\n";
                        op(geo.begin(), geo.end(), curvature.begin(), closedLine);

                        if(false){
                            //std::cout<<"    fill histogram"<<"\n";
                            // clear histogram -> set min max from data and fill from data
                            histogram.clearSetMinMaxAndFillFrom(curvature.begin(), curvature.begin()+geo.size());

                            //std::cout<<"    get quantiles"<<"\n";
                            // fetch the quantiles
                            nifty::histogram::quantiles(histogram, quantiles_.begin(), quantiles_.end(),
                                quantilesOut.begin());

                            //std::cout<<"    write to features"<<"\n";
                            // yay we are done for this cell1
                            // => just write results to features
                            const auto fIndex = sigmaIndex;//*quantiles_.size();
                            for(auto qIndex=0; qIndex<quantiles_.size(); ++qIndex){
                                features(cell1Index, fIndex + qIndex) = quantilesOut[qIndex];
                            }
                        }
                        else{
                            features(cell1Index, sigmaIndex) = std::accumulate(
                                curvature.begin(), curvature.begin()+geo.size(),0.0
                            )/geo.size();
                        }
                    }
                    else{
                        if(false){
                            const auto fIndex = sigmaIndex*quantiles_.size();
                            for(auto qIndex=0; qIndex<quantiles_.size(); ++qIndex){
                                features(cell1Index, fIndex + qIndex) = 0.0;
                            }
                        }
                        else{
                            features(cell1Index, sigmaIndex) = 0.0;
                        }   
                    }
                }
            }
        }   
    private:
        std::vector<float> sigmas_;
        std::vector<float> quantiles_;
    };




    class Cell1LineSegmentDist2D{
    public:
        Cell1LineSegmentDist2D(
            const std::vector<size_t> & dists  = std::vector<size_t>({size_t(3),size_t(5),size_t(7)})
        )
        :   dists_(dists)
        {
        }

        size_t numberOfFeatures()const{
            return  dists_.size();
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,1>  & cell1GeometryVector,
            nifty::marray::View<T> & features
        )const{  

            

            typedef boost::geometry::model::d2::point_xy<double> point_type;
            //typedef boost::geometry::model::polygon<point_type> polygon_type;
            typedef boost::geometry::model::linestring<point_type> linestring_type;
            //typedef boost::geometry::model::multi_point<point_type> multi_point_type;

            for(auto di=0; di<dists_.size(); ++di){
                const auto ld = dists_[di];
                for(auto cell1Index=0; cell1Index<cell1GeometryVector.size(); ++cell1Index){
                    const auto & geo = cell1GeometryVector[cell1Index];
                    if(geo.size()>=4){
                        auto d = 0.0;
                        for(auto i=0; i<geo.size()-1; ++i){
                            const auto j = std::min(int(i + ld), int(geo.size()-1));
                            const auto & pS = geo[i];
                            const auto & pE = geo[j];   

                            linestring_type line;
                            line.push_back(point_type(pS[0], pS[1]));
                            line.push_back(point_type(pE[0], pE[1]));
                            
                            for(auto ii=i+1; ii<j-1; ++ii){
                                const point_type p(geo[ii][0], geo[ii][1]);
                                d += boost::geometry::distance(p, line);
                                //std::cout<<"d "<<d<<"\n";
                            }
                        }
                        features(cell1Index,di) = d/float(geo.size());
                    }
                    else{
                        features(cell1Index,di) = 0.0;
                    }
                }
            }
        }
    private:
        std::vector<size_t> dists_;
    };


    class Cell1BasicGeometricFeatures2D{
    public:
        Cell1BasicGeometricFeatures2D(
            const std::vector<size_t> & dists  = std::vector<size_t>({size_t(3),size_t(5),size_t(7)})
        )
        :   dists_(dists)
        {
        }

        size_t numberOfFeatures()const{
            return 20;
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,0>   & cell0GeometryVector,
            const CellGeometryVector<2,1>   & cell1GeometryVector,
            const CellGeometryVector<2,2>   & cell2GeometryVector,
            const CellBoundsVector<2,0>     & cell0BoundsVector,
            const CellBoundsVector<2,1>     & cell1BoundsVector,
            const CellBoundedByVector<2,1>  & cell1BoundedByVector,
            const CellBoundedByVector<2,2>  & cell2BoundedByVector,
            nifty::marray::View<T> & features
        )const{  
            using namespace nifty::math;
            for(auto cell1Index=0; cell1Index<cell1GeometryVector.size(); ++cell1Index){

                const auto & cell1Bounds = cell1BoundsVector[cell1Index];
                const auto cell2UIndex = cell1Bounds[0]-1;
                const auto cell2VIndex = cell1Bounds[1]-1;


                auto fIndex = 0;

                auto insertCell2ValFeats = [&](const float uVal, const float vVal){
                    features(cell1Index, fIndex++) = std::min(uVal, vVal);
                    features(cell1Index, fIndex++) = std::max(uVal, vVal);
                    features(cell1Index, fIndex++) = uVal + vVal;
                    features(cell1Index, fIndex++) = std::abs(uVal-vVal);
                };

                const auto & geoE = cell1GeometryVector[cell1Index];
                const auto & geoU = cell2GeometryVector[cell2UIndex];
                const auto & geoV = cell2GeometryVector[cell2VIndex];

                // size based features
                const auto eSize = float(geoE.size());
                const auto uSize = float(geoU.size());
                const auto vSize = float(geoV.size());
                   

                features(cell1Index, fIndex++) = eSize; 
                insertCell2ValFeats(uSize, vSize);

                // size ratios
                const auto uNSize = std::sqrt(uSize);
                const auto vNSize = std::sqrt(vSize);  
                {
                    const auto ratU =  uNSize/eSize;
                    const auto ratV =  vNSize/eSize;  
                    insertCell2ValFeats(ratU, ratV);
                }


                // endpoint distance and ratios
                const auto endpointDistance = euclideanDistance(geoE.front(), geoE.back());
                features(cell1Index, fIndex++) = endpointDistance;
                features(cell1Index, fIndex++) = endpointDistance/eSize;

                // distance between cell2CenterOfMass  and cell1CenterOfMass
                const auto  comE = geoE.centerOfMass();
                const auto  comU = geoU.centerOfMass();
                const auto  comV = geoV.centerOfMass();
                {
                    const auto dUV  = euclideanDistance(comU, comV);
                    const auto dUE  = euclideanDistance(comU, comE);
                    const auto dVE  = euclideanDistance(comV, comE);

                    features(cell1Index, fIndex++) = dUV;
                    insertCell2ValFeats(dUE, dVE);
                

                    // distance ratios between cell2CenterOfMass  and cell1CenterOfMass
                
                    const auto ratU =  uNSize/dUE;
                    const auto ratV =  vNSize/dVE;  
                    insertCell2ValFeats(ratU, ratV);
                }


                // angle between cell2CenterOfMass  and cell1CenterOfMass
                // ...
                


            }
        
        }
    private:
        std::vector<size_t> dists_;
    };




    class Cell1Cell2CurvatureFeatures2D{
    };





    /*    
     *********************************************
     *    
     *    WHAT ABOUT ORIENTATION AND ANGLES???:  
     *   
     *********************************************    
     *
     * 
     */




















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