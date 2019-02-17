#pragma once

#include "nifty/math/numerics.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace filters{



template<class T = long double>
class GaussianCurvature2D{

public:
    typedef typename nifty::math::NumericTraits<T>::RealPromote ValueType;
    typedef ValueType value_type;

    GaussianCurvature2D(const ValueType sigma, int r = -1, const ValueType windowRatio = 3.5)
    :   sigma_(sigma),
        radius_(std::max(1,r == -1 ? int(sigma*windowRatio + 0.5) : r    )),
        kdx_(radius_*2 + 1),
        kdxx_(radius_*2 + 1),
        eps_(0.000000001)
    {
        // init the kernels
        const auto sigmaP2 = sigma*sigma;
        const auto sigmaP4 = sigmaP2*sigmaP2;
        ValueType sx = 0;
        ValueType sxx = 0;
        for(int i=0; i<kdx_.size(); ++i){
            const auto x = double(i - radius_);
            const auto xP2 = x*x;
            const auto arg = -1.0 * xP2/(sigmaP2);
            const auto g0 = std::exp(arg);
            const auto g1 = -1.0*(-1.0*x/sigmaP2) * g0;
            const auto g2 = ((xP2 - sigmaP2)/sigmaP4 ) * g0;
            kdx_[i] = g1;
            kdxx_[i] = g2;
            sx += g1;
            sxx += g2;
        }
        for(int i=0; i<kdx_.size(); ++i){
            kdxx_[i] -= (sxx / kdx_.size());
        }
    }


    template<class ARR0, class ARR1>
    void operator()(const ARR0 & coordinates,
                    ARR1 & out,
                    const bool closedLine) const {
        //
        typedef typename ARR0::value_type T0;
        typedef typename ARR1::value_type T1;

        struct CoordAdaptor{
            CoordAdaptor(const ARR0 & _coordinates)
            : coordinates_(_coordinates){
            }
            //
            std::array<ValueType, 2> operator[](const std::size_t i)const{
                std::array<ValueType,2> ret;
                ret[0] = coordinates_(i, 0);
                ret[1] = coordinates_(i, 1);
                return ret;
            }
            const ARR0 & coordinates_;
        };

        struct OutAdaptor{
            OutAdaptor(ARR1 & _out)
            : out_(_out){
            }
            T1 & operator[](const std::size_t i) {
                return out_(i);
            }
            ARR1 & out_;
        };
        this->impl(CoordAdaptor(coordinates), OutAdaptor(out),
                    coordinates.shape()[0], closedLine);
    }

    template<class COORD_ITER, class OUT_ITER>
    void operator()(
        COORD_ITER coordsBegin,
        COORD_ITER coordsEnd,
        OUT_ITER   outIter,
        const bool closedLine
    ) const {
        this->impl(coordsBegin, outIter,
            std::distance(coordsBegin, coordsEnd),
            closedLine);
    }


    int radius()const{
        return radius_;
    }
private:

    template<class COORD_ITER, class OUT_ITER>
    void impl(
        COORD_ITER coordsBegin,
        OUT_ITER   outIter,
        const std::size_t size,
        const bool closedLine
    ) const {

        const auto kSize = 2*radius_ + 1;

        struct CoordinateExtrapolator{
            CoordinateExtrapolator(
                const COORD_ITER & _coordinates,
                const std::size_t s
            ):  coordinates_(_coordinates),
                size_(s)
            {

                for(auto d=0; d<2; ++d){
                    dLow_[d] = coordinates_[0][d] - coordinates_[1][d];
                    dHigh_[d] = coordinates_[s-1][d] - coordinates_[s-2][d];
                }
            }

            std::array<ValueType,2> operator[](const int i)const{
                std::array<ValueType,2>  ret;
                const int s = size_;
                if(i >= 0 && i < s){
                    const auto c =  coordinates_[i];
                    ret[0] = c[0];
                    ret[1] = c[1];
                }
                else if(i < 0){
                    const auto c =  coordinates_[0];
                    const auto d = ValueType(std::abs(i));
                    ret[0] = ValueType(c[0]) + dLow_[0]*d;
                    ret[1] = ValueType(c[1]) + dLow_[1]*d;
                }
                else{ // i >= size
                    const auto c =  coordinates_[s-1];
                    const auto d = ValueType(i - (s - 1));
                    ret[0] = ValueType(c[0]) + dHigh_[0]*d;
                    ret[1] = ValueType(c[1]) + dHigh_[1]*d;
                }
                return ret;
            }

            const COORD_ITER & coordinates_;
            std::size_t size_;
            std::array<ValueType,2>  dLow_;
            std::array<ValueType,2>  dHigh_;
        };

        CoordinateExtrapolator cExt(coordsBegin, size);

        for(int i=0; i<size; ++i){

            ValueType dx[2]  = {ValueType(0),ValueType(0)};
            ValueType dxx[2] = {ValueType(0),ValueType(0)};

            for(int ki=0; ki<kSize; ++ki){

                const auto di = i - radius_ + ki;
                const auto eCoord = cExt[di];

                for(auto d=0; d<2; ++d){
                    dx[d]  += ValueType(eCoord[d]) * kdx_[ki];
                    dxx[d] += ValueType(eCoord[d]) * kdxx_[ki];
                }
            }

            const auto a = std::abs(dx[0]*dxx[1] -  dx[1]*dxx[0]);
            const auto b = std::pow(dx[0]*dx[0] + dx[1]*dx[1], 3.0/2.0);


            if(std::abs(a) < eps_ && std::abs(b) < eps_){
                outIter[i] = 0.0;
            }
            else{
                const auto k  = (a)/(b);
                outIter[i] = k;
            }
        }
    }



    ValueType sigma_;
    int radius_;
    std::vector<ValueType> kdx_;
    std::vector<ValueType> kdxx_;
    ValueType eps_;
};




#if 0

template<class T = long double>
class StepCurvature2D{

public:
    typedef typename nifty::math::NumericTraits<T>::RealPromote ValueType;
    typedef ValueType value_type;

    StepCurvature2D(const int step = 2)
    :   step_(step)
    {

    }


    template<class T0, class T1>
    void operator()(
        const xt::xtensor<T0, 2> & coordinates,
        xt::xtensor<T1, 1> & out
    ) const {
        int step  = step_;
        std::vector< double > vecCurvature( vecContourPoints.size() );
        if (vecContourPoints.size() < step){
            step = 1;
        }

        bool isClosed = false;
        cv::Point2f pplus, pminus;
        cv::Point2f f1stDerivative, f2ndDerivative;
        for (int i = 0; i < vecContourPoints.size(); i++ ){
            const cv::Point2f& pos = vecContourPoints[i];

            int maxStep = step;
            if (!isClosed){
                maxStep = std::min(std::min(step, i), (int)vecContourPoints.size()-1-i);
                if (maxStep == 0){
                    vecCurvature[i] = std::numeric_limits<double>::infinity();
                    continue;
                }
            }

            int iminus = i-maxStep;
            int iplus = i+maxStep;
            pminus = vecContourPoints[iminus < 0 ? iminus + vecContourPoints.size() : iminus];
            pplus = vecContourPoints[iplus > vecContourPoints.size() ? iplus - vecContourPoints.size() : iplus];


            f1stDerivative.x =   (pplus.x -        pminus.x) / (iplus-iminus);
            f1stDerivative.y =   (pplus.y -        pminus.y) / (iplus-iminus);
            f2ndDerivative.x = (pplus.x - 2*pos.x + pminus.x) / ((iplus-iminus)/2*(iplus-iminus)/2);
            f2ndDerivative.y = (pplus.y - 2*pos.y + pminus.y) / ((iplus-iminus)/2*(iplus-iminus)/2);

            double curvature2D;
            double divisor = f1stDerivative.x*f1stDerivative.x + f1stDerivative.y*f1stDerivative.y;
            if ( std::abs(divisor) > 10e-8 ){
                curvature2D =  std::abs(f2ndDerivative.y*f1stDerivative.x - f2ndDerivative.x*f1stDerivative.y) /
                        pow(divisor, 3.0/2.0 )  ;
            }
            else{
                curvature2D = std::numeric_limits<double>::infinity();
            }
            vecCurvature[i] = curvature2D;
        }
        return vecCurvature;

    }


    int step()const{
        return step_;
    }
private:
    int step_;
    std::vector<ValueType> kdx_;
    std::vector<ValueType> kdxx_;
};
#endif

} // end namespace nifty::filters
} // end namespace nifty
