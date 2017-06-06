#pragma once

#include <vector>

#include "nifty/math/math.hxx"
#include "nifty/histogram/histogram.hxx"
#include "nifty/cgp/geometry.hxx"
#include "nifty/cgp/bounds.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/filters/gaussian_curvature.hxx"
#include "nifty/features/accumulated_features.hxx"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>


namespace nifty{
namespace cgp{

    class Cell1CurvatureFeatures2D{
    private:
        typedef nifty::features::DefaultAccumulatedStatistics<float> AccType;
    public:
        Cell1CurvatureFeatures2D(
            const std::vector<float> & sigmas  = std::vector<float>({1.0f, 2.0f, 4.0f})
        )
        :   sigmas_(sigmas)
        {

        }

        size_t numberOfFeatures()const{
            return sigmas_.size() * AccType::NFeatures::value;   
        }
        

        std::vector<std::string> names()const{

            std::string accNames [] = {
                std::string("Mean"),
                std::string("Sum"),
                std::string("Min"),
                std::string("Max"),
                std::string("Moment2"),
                std::string("Moment3"),
                std::string("Q0.10"),
                std::string("Q0.25"),
                std::string("Q0.50"),
                std::string("Q0.75"),
                std::string("Q0.90")
            };
            std::vector<std::string> res;
            auto baseName = std::string("GaussianCurvatureSigma");
            for(auto sigmaIndex=0; sigmaIndex<sigmas_.size(); ++sigmaIndex){
                auto name  = baseName + std::to_string(sigmas_[sigmaIndex]);
                for(const auto & accName : accNames){    
                    res.push_back(name + accName);
                }
                
            }
            return res;
        }

        template<class T>
        void operator()(
            const CellGeometryVector<2,1>  & cell1GeometryVector,
            const CellBoundedByVector<2,1> & cell1BoundedByVector,
            nifty::marray::View<T> & features
        )const{  

            std::vector<float> curvature;
            std::vector<float> buffer(AccType::NFeatures::value);
            

            
            for(auto sigmaIndex=0; sigmaIndex<sigmas_.size(); ++sigmaIndex){
                const auto sigma = sigmas_[sigmaIndex];
                nifty::filters::GaussianCurvature2D<> op(sigma, -1, 2.5);


                for(auto cell1Index=0; cell1Index<cell1GeometryVector.size(); ++cell1Index){
                    const auto & geo = cell1GeometryVector[cell1Index];

                    

                    if(geo.size()>=4){

               
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

                        // accumulate the values
                        AccType acc;
                        for(auto pass=0; pass < acc.requiredPasses(); ++pass){
                            for(const auto & c : curvature){
                                acc.acc(c, pass);
                            }
                        }
                        // write to buffer
                        acc.result(buffer.begin(), buffer.end());   
                        
                        // write to features out
                        const auto fIndex = sigmaIndex*AccType::NFeatures::value;
                        for(auto afi=0; afi<AccType::NFeatures::value; ++afi){
                            features(cell1Index, fIndex + afi) = buffer[afi];
                        }

                        
                    }
                    else{
                        // write to features out
                        // a zero value seems legit since we
                        // assume a constant zero curvature
                        const auto fIndex = sigmaIndex*AccType::NFeatures::value;
                        for(auto afi=0; afi<AccType::NFeatures::value; ++afi){
                            features(cell1Index, fIndex + afi) = 0.0;
                        }

                    }
                }
            }
        }   
    private:
        std::vector<float> sigmas_;
    };




    class Cell1LineSegmentDist2D{
    private:
        typedef nifty::features::DefaultAccumulatedStatistics<float> AccType;
    public:
        Cell1LineSegmentDist2D(
            const std::vector<size_t> & dists  = std::vector<size_t>({size_t(3),size_t(5),size_t(7)})
        )
        :   dists_(dists)
        {
        }

        size_t numberOfFeatures()const{
            return  dists_.size()*AccType::NFeatures::value;
        }

       std::vector<std::string> names()const{

            std::string accNames [] = {
                std::string("Mean"),
                std::string("Sum"),
                std::string("Min"),
                std::string("Max"),
                std::string("Moment2"),
                std::string("Moment3"),
                std::string("Q0.10"),
                std::string("Q0.25"),
                std::string("Q0.50"),
                std::string("Q0.75"),
                std::string("Q0.90")
            };
            std::vector<std::string> res;
            auto baseName = std::string("Cell1LineSegmentDistD");
            for(auto i=0; i<dists_.size(); ++i){
                auto name  = baseName + std::to_string(dists_[i]);
                for(const auto & accName : accNames){
                    res.push_back(name + accName);
                }
                
            }
            return res;
        }



        template<class T>
        void operator()(
            const CellGeometryVector<2,1>  & cell1GeometryVector,
            nifty::marray::View<T> & features
        )const{  

            std::vector<float> buffer(AccType::NFeatures::value);

            typedef boost::geometry::model::d2::point_xy<double> point_type;
            typedef boost::geometry::model::linestring<point_type> linestring_type;


            for(auto di=0; di<dists_.size(); ++di){
                const auto ld = dists_[di];
                for(auto cell1Index=0; cell1Index<cell1GeometryVector.size(); ++cell1Index){
                    const auto & geo = cell1GeometryVector[cell1Index];
                    if(geo.size()>=4){

                        AccType acc;

                        for(auto pass=0; pass < acc.requiredPasses(); ++pass){

                            for(auto i=0; i<geo.size()-1; ++i){
                                const auto j = std::min(int(i + ld), int(geo.size()-1));
                                const auto & pS = geo[i];
                                const auto & pE = geo[j];   

                                linestring_type line;
                                line.push_back(point_type(pS[0], pS[1]));
                                line.push_back(point_type(pE[0], pE[1]));
                                
                                for(auto ii=i+1; ii<j-1; ++ii){
                                    const point_type p(geo[ii][0], geo[ii][1]);
                                    const auto d = boost::geometry::distance(p, line);
                                    acc.acc(d, pass);
                                }
                            }
                        }
                        // write to buffer
                        acc.result(buffer.begin(), buffer.end());

                        // write to features out
                        const auto fIndex = di*AccType::NFeatures::value;
                        for(auto afi=0; afi<AccType::NFeatures::value; ++afi){
                            features(cell1Index, fIndex + afi) = buffer[afi];
                        }

                        
                    }
                    else{
                        // write to features out
                        // a zero value seems legit since we
                        // assume a constant zero curvature
                        const auto fIndex = di*AccType::NFeatures::value;
                        for(auto afi=0; afi<AccType::NFeatures::value; ++afi){
                            features(cell1Index, fIndex + afi) = 0.0;
                        }
                    }
                }
            }
        }
    private:
        std::vector<size_t> dists_;
    };


    class Cell1BasicGeometricFeatures2D{
    private:
        typedef nifty::features::DefaultAccumulatedStatistics<float> AccType;
    public:
        Cell1BasicGeometricFeatures2D(){
        }

        size_t numberOfFeatures()const{
            return 4 * 4 +  2*AccType::NFeatures::value + 4;   
        }



        std::vector<std::string> names()const{

            std::string accNames [] = {
                std::string("Mean"),
                std::string("Sum"),
                std::string("Min"),
                std::string("Max"),
                std::string("Moment2"),
                std::string("Moment3"),
                std::string("Q0.10"),
                std::string("Q0.25"),
                std::string("Q0.50"),
                std::string("Q0.75"),
                std::string("Q0.90")
            };

            std::vector<std::string> res;
            const auto baseName = std::string("BasicGeometricFeatures");
            auto insertUVFeat = [&](const std::string & name){
                res.push_back(baseName+name+std::string("UV-Min"));
                res.push_back(baseName+name+std::string("UV-Max"));
                res.push_back(baseName+name+std::string("UV-Sum"));
                res.push_back(baseName+name+std::string("UV-AbsDiff"));
            };
            auto insertStatFeat = [&](const std::string & name){
                for(const auto & an : accNames){
                    res.push_back(baseName+name+an);
                }
            };

            res.push_back(baseName+std::string("EdgeSize"));
            insertUVFeat("NodeSize");
            insertUVFeat("NodeEdgeSizeRatio");

            res.push_back(baseName+std::string("EdgeEndpointDistance"));
            res.push_back(baseName+std::string("EdgeRelativeEndpointDistance"));
            res.push_back(baseName+std::string("NodeEndpointDistance"));

            
            insertUVFeat("EdgeNodeCenterOfMassDist");
            insertUVFeat("EdgeNodeCenterOfMassRatio");  

            insertStatFeat("CenterOfMassCell1PointsDist");
            insertStatFeat("CenterOfMassCell2PointsDist");

            return res;
        }

        template<class T>
        void operator()(
            const CellGeometryVector<2,1>   & cell1GeometryVector,
            const CellGeometryVector<2,2>   & cell2GeometryVector,
            const CellBoundsVector<2,1>     & cell1BoundsVector,
            nifty::marray::View<T> & features
        )const{  

            using namespace nifty::math;
            std::vector<float> buffer(AccType::NFeatures::value);
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

                // statistic  over ||centerOfMass-points||  for cell 1
                {
                    AccType acc;
                    for(auto i=0; i<geoE.size(); ++i){
                        const auto & coord = geoE[i];
                        const auto d  = nifty::math::euclideanDistance(coord, comE);
                        acc.acc(d);
                    }
                    // write to buffer
                    acc.result(buffer.begin(), buffer.end()); 
                    // write results
                    for(auto i=0; i<buffer.size(); ++i){
                        features(cell1Index,  fIndex++) = buffer[i];
                    }
                }





                // statistic  over ||centerOfMass_cell1-points_cell2|| 
                {
                    AccType acc;


                    for(auto i=0; i<geoU.size(); ++i){
                        const auto & coord = geoU[i];
                        const auto d  = nifty::math::euclideanDistance(coord, comE);
                        acc.acc(d);
                    }
                    for(auto i=0; i<geoV.size(); ++i){
                        const auto & coord = geoV[i];
                        const auto d  = nifty::math::euclideanDistance(coord, comE);
                        acc.acc(d);
                    }

                    // write to buffer and write results
                    acc.result(buffer.begin(), buffer.end()); 

                    for(auto i=0; i<buffer.size(); ++i){
                        features(cell1Index,  fIndex++) = buffer[i];
                    }

                }


                // angle between cell2CenterOfMass  and cell1CenterOfMass
                // ...



                NIFTY_CHECK_OP(fIndex,==,numberOfFeatures(),"internal error");


            }
        
        }
    private:
        std::vector<size_t> dists_;
    };


    /**
    class GeometricAccumulator{
    private:
        typedef nifty::features::DefaultAccumulatedStatistics<float> AccType;
    public:
        GeometricAccumulator(
            const std::vector<size_t> & dists  = std::vector<size_t>({size_t(3),size_t(5),size_t(7)})
        )
        :   dists_(dists)
        {
        }

        size_t numberOfFeatures()const{
            return  dists_.size()*AccType::NFeatures::value;
        }
            

        template<class T>
        void operator()(
            const CellGeometryVector<2,1>  & cell1GeometryVector,
            nifty::marray::View<T> & features
        )const{  

            std::vector<float> buffer(AccType::NFeatures::value);

            typedef boost::geometry::model::d2::point_xy<double> point_type;
            typedef boost::geometry::model::linestring<point_type> linestring_type;


                
            for(auto cell1Index=0; cell1Index<cell1GeometryVector.size(); ++cell1Index){

                const auto & geo = cell1GeometryVector[cell1Index];
                
            }
        
        }
    private:
        std::vector<size_t> dists_;
    };
    */



}
}