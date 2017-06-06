#pragma once

#include <mutex>
#include <map>
#include <cmath>

#include "fastfilters.h"

#include "nifty/marray/marray.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/array/arithmetic_array.hxx"

namespace nifty{
namespace features{

namespace detail_fastfilters {
    
    // copied from fastfilters/python/core.hxx
    template <typename fastfilters_array_t> struct FastfiltersDim {
    };
    
    template <> struct FastfiltersDim<fastfilters_array2d_t> {
        static const unsigned int ndim = 2;
    
        static void set_z(size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
        {
        }
        static void set_stride_z(size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
        {
        }
    };
    
    template <> struct FastfiltersDim<fastfilters_array3d_t> {
        static const unsigned int ndim = 3;
    
        static void set_z(size_t n_z, fastfilters_array3d_t &k)
        {
            k.n_z = n_z;
        }
        static void set_stride_z(size_t stride_z, fastfilters_array3d_t &k)
        {
            k.stride_z = stride_z;
        }
    };

    
    // adapted from fastfilters/python/core.hxx, convert_py2ff
    template <typename fastfilters_array_t>
    void convertMarray2ff(const marray::View<float> & array, fastfilters_array_t & ff) {
        
        const unsigned int dim = FastfiltersDim<fastfilters_array_t>::ndim;
    
        if (array.dimension() >= (int) dim) {
            ff.ptr = (float *) &array(0);
    
            ff.n_x = array.shape(dim - 1);
            ff.stride_x = array.strides(dim - 1);
    
            ff.n_y = array.shape(dim - 2);
            ff.stride_y = array.strides(dim - 2);
    
            if (dim == 3) {
                FastfiltersDim<fastfilters_array_t>::set_z(array.shape(dim - 3), ff);
                FastfiltersDim<fastfilters_array_t>::set_stride_z(array.strides(dim - 3), ff);
            }
        } else {
            throw std::runtime_error("Too few dimensions.");
        }
    
        if (array.dimension() == dim) {
            ff.n_channels = 1;
        } 
        //else if ((array.dimension() == dim + 1) && array.shape(dim) < 8 && array.strides(dim) == sizeof(float)) {
        else if ((array.dimension() == dim + 1) && array.shape(dim) < 8 && array.strides(dim) == 1) {
            ff.n_channels = array.shape(dim);
        } else {
            throw std::runtime_error("Invalid number of dimensions or too many channels or stride between channels.");
        }
    }
} //namespace detail_fastfilters

    //
    // Functors for the individual filters
    //

    // flag for call_once (don't know if this would be thread safe as (static) member)    
    std::once_flag onceFlag;

    struct FilterBase {
        
        FilterBase() {
            std::call_once(onceFlag, []() {
                fastfilters_init();
            });
            opt_.window_ratio = 0.;
        }

        virtual void operator()(const fastfilters_array2d_t &, marray::View<float> &, const double) const = 0;

        virtual void operator()(const fastfilters_array3d_t &, marray::View<float> &, const  double) const = 0; 

        virtual bool isMultiChannel() const = 0;

        void set_window_ratio(const double ratio) {
            opt_.window_ratio = ratio;
        }

    protected:
        fastfilters_options_t opt_;
    };

    struct GaussianSmoothing : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gaussian2d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianSmoothing 2d failed.");
        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gaussian3d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianSmoothing 3d failed.");
        }

        bool isMultiChannel() const {
            return false;
        }

    };
    
    struct LaplacianOfGaussian : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_laplacian2d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("LaplacianOfGaussian 2d failed!");
        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_laplacian3d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("LaplacianOfGaussian 3d failed!");
        }
        
        bool inline isMultiChannel() const {
            return false;
        }

    };
    

    struct GaussianGradientMagnitude : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gradmag2d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianGradientMagnitude 2d failed!");
        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gradmag3d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianGradientMagnitude 3d failed!");
        }
        
        bool inline isMultiChannel() const {
            return false;
        }
        

    };
    
    struct HessianOfGaussianEigenvalues : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {
            
            fastfilters_array2d_t * xx = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * yy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * xy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);

            if( !fastfilters_fir_hog2d(&ff, sigma, xx, xy, yy, &opt_) ) 
                throw std::runtime_error("HessianOfGaussian 2d failed.");

            const size_t numberOfPixels = ff.n_x * ff.n_y;

            float* ev0 = &out(0);
            float* ev1 = &out(0) + numberOfPixels;

            fastfilters_linalg_ev2d(xx->ptr, xy->ptr, yy->ptr, ev0, ev1, numberOfPixels);

            fastfilters_array2d_free(xx);
            fastfilters_array2d_free(yy);
            fastfilters_array2d_free(xy);

        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array3d_t * xx = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * zz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);

            if( !fastfilters_fir_hog3d(&ff, sigma, xx, yy, zz, xy, xz, yz, &opt_) ) 
                throw std::runtime_error("HessianOfGaussian 3d failed.");

            const size_t numberOfPixels = ff.n_x * ff.n_y * ff.n_z;

            float* ev0 = &out(0);
            float* ev1 = &out(0) + numberOfPixels;
            float* ev2 = &out(0) + 2*numberOfPixels;

            fastfilters_linalg_ev3d(zz->ptr, yz->ptr, xz->ptr, yy->ptr, xy->ptr, xx->ptr, ev0, ev1, ev2, numberOfPixels);
            
            fastfilters_array3d_free(xx);
            fastfilters_array3d_free(yy);
            fastfilters_array3d_free(zz);
            fastfilters_array3d_free(xy);
            fastfilters_array3d_free(xz);
            fastfilters_array3d_free(yz);

        }
        
        bool inline isMultiChannel() const {
            return true;
        }

    };
    

    // outer scale has to be set as member variable to keep consistency w/ the operator()
    struct StructureTensorEigenvalues : FilterBase {

        StructureTensorEigenvalues() : FilterBase(), sigmaOuter_(0.) {
        }
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {

            NIFTY_CHECK_OP(sigma,<,sigmaOuter_,"inner scale has to be smaller than outer scale, set via setOuterScale")
            
            fastfilters_array2d_t * xx = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * yy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * xy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);

            if( !fastfilters_fir_structure_tensor2d(&ff, sigma, sigmaOuter_, xx, xy, yy, &opt_) ) 
                throw std::runtime_error("StructurTensor 2d failed.");

            const size_t numberOfPixels = ff.n_x * ff.n_y;

            float* ev0 = &out(0);
            float* ev1 = &out(0) + numberOfPixels;

            fastfilters_linalg_ev2d(xx->ptr, xy->ptr, yy->ptr, ev0, ev1, numberOfPixels);

            fastfilters_array2d_free(xx);
            fastfilters_array2d_free(yy);
            fastfilters_array2d_free(xy);

        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
            
            NIFTY_CHECK_OP(sigma,<,sigmaOuter_,"inner scale has to be smaller than outer scale, set via setOuterScale")
            
            fastfilters_array3d_t * xx = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * zz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);

            if( !fastfilters_fir_structure_tensor3d(&ff, sigma, 2*sigma, xx, yy, zz, xy, xz, yz, &opt_) ) 
                throw std::runtime_error("StructureTensor 3d failed.");

            const size_t numberOfPixels = ff.n_x * ff.n_y * ff.n_z;

            float* ev0 = &out(0);
            float* ev1 = &out(0) + numberOfPixels;
            float* ev2 = &out(0) + 2*numberOfPixels;

            fastfilters_linalg_ev3d(zz->ptr, yz->ptr, xz->ptr, yy->ptr, xy->ptr, xx->ptr, ev0, ev1, ev2, numberOfPixels);
            
            fastfilters_array3d_free(xx);
            fastfilters_array3d_free(yy);
            fastfilters_array3d_free(zz);
            fastfilters_array3d_free(xy);
            fastfilters_array3d_free(xz);
            fastfilters_array3d_free(yz);

        }
        
        bool inline isMultiChannel() const {
            return true;
        }

        void inline setOuterScale(const double sigmaOuter) {
            sigmaOuter_ = sigmaOuter;
        }

    private:
        double sigmaOuter_;

    };

    template<unsigned DIM> 
    struct ApplyFilters {
        
        typedef typename std::conditional<DIM==2,
            fastfilters_array2d_t, fastfilters_array3d_t >::type
        FastfiltersArrayType;
        typedef array::StaticArray<int64_t, DIM+1> Coord;
        // enum encoding the different filters FIXME no structure tensor for now due to second scale
        enum filterNames {GaussianSmoothing = 0, LaplacianOfGaussian = 1, GaussianGradientMagnitude = 2, HessianOfGaussianEigenvalues = 3};
        typedef std::vector<std::vector<bool>> FiltersToSigmasType;
        
        ApplyFilters(const std::vector<double> & sigmas,
                const FiltersToSigmasType & filtersToSigmas) 
            : sigmas_(sigmas),
              filtersToSigmas_(filtersToSigmas){
            
            // initialize the filters according to filtersToSigmas
            NIFTY_CHECK_OP(filtersToSigmas_.size(),==,4,"The size of filters to sigmas must correspond to the number of possible filters.");
            for(const auto & filtToSig : filtersToSigmas)
                NIFTY_CHECK_OP(filtToSig.size(),==,sigmas_.size(),"");
            
            for(size_t ii = 0; ii < filtersToSigmas_.size(); ++ii) {
                // if at least one sigma is active, we push back the corresponding filter
                const auto & filtToSig = filtersToSigmas_[ii];
                bool hasSigma = false;
                for(auto val : filtToSig) {
                    if(val)
                        hasSigma = true;
                }
                activeFilters_[ii] = hasSigma;
                if(hasSigma) {
                    switch(ii) {
                        case 0: filters_.emplace(std::make_pair(ii, new nifty::features::GaussianSmoothing)); break;
                        case 1: filters_.emplace(std::make_pair(ii, new nifty::features::LaplacianOfGaussian)); break;
                        case 2: filters_.emplace(std::make_pair(ii, new nifty::features::GaussianGradientMagnitude)); break;
                        case 3: filters_.emplace(std::make_pair(ii, new nifty::features::HessianOfGaussianEigenvalues)); break;
                        dfault: throw std::runtime_error("Invalid filter selection!");
                    }
                }
            }
        }

        ~ApplyFilters(){
            for(auto filterIt = filters_.begin(); filterIt != filters_.end(); ++filterIt)
                delete filterIt->second;
        }
        
        // apply filters sequential
        void operator()(const marray::View<float> & in, marray::View<float> & out, const bool presmooth = false) const{
            Coord shapeSingleChannel, shapeMultiChannel, base;
            applyCommon(in, out, shapeSingleChannel, shapeMultiChannel, base);
            if(presmooth)
                applyFiltersSequentialWithPresmoothing(in, out, shapeSingleChannel, shapeMultiChannel, base);
            else
                applyFiltersSequential(in, out, shapeSingleChannel, shapeMultiChannel, base);
        }
        
        // apply filters in parallel
        void operator()(const marray::View<float> & in, marray::View<float> & out, parallel::ThreadPool & threadpool) const{
            Coord shapeSingleChannel, shapeMultiChannel, base;
            applyCommon(in, out, shapeSingleChannel, shapeMultiChannel, base);
            applyFiltersParallel(in, out, shapeSingleChannel, shapeMultiChannel, base, threadpool);
        }

        size_t numberOfChannels() const {
            size_t numberOfChannels = 0;
            for( size_t ii = 0; ii < activeFilters_.size(); ++ii ) {
                if( !activeFilters_[ii] )
                    continue;
                size_t nChans = filters_.at(ii)->isMultiChannel() ? DIM : 1;
                const auto & activeSigmas = filtersToSigmas_[ii];
                for( size_t jj = 0; jj < sigmas_.size(); ++jj ) {
                    if( activeSigmas[jj] )
                        numberOfChannels += nChans;
                }
            }     
            return numberOfChannels;
        }
            
    private:
        
        // common pre-processing for all overloads of operator()
        inline void applyCommon(const marray::View<float> & in, 
                marray::View<float> & out,
                Coord & shapeSingleChannel,
                Coord & shapeMultiChannel,
                Coord & base) const{
            // checks
            NIFTY_CHECK_OP(in.dimension(),==,DIM,"Input needs to be of correct dimension.")
            NIFTY_CHECK_OP(out.shape(0),==,numberOfChannels(),"Number of channels of out array do not match!")
            for(int d = 0; d < DIM; ++d)
                NIFTY_CHECK_OP(out.shape(d+1),==,in.shape(d),"In and out axis do not agree!")
            // determine shapes
            shapeSingleChannel[0] = 1L;
            shapeMultiChannel[0] = int64_t(DIM);
            base[0] = 0L;
            for(int d = 0; d < DIM; d++) {
               shapeSingleChannel[d+1] = in.shape(d); 
               shapeMultiChannel[d+1] = in.shape(d); 
               base[d+1] = 0L;
            }
        }
            
        inline void applyFiltersSequential(const marray::View<float> & in,
                marray::View<float> & out,
                const Coord & shapeSingleChannel,
                const Coord & shapeMultiChannel,
                Coord & base ) const {
            
            // copy in-marray to fastfilters array
            FastfiltersArrayType ff;
            detail_fastfilters::convertMarray2ff(in, ff);

            // apply filters sequentially
            for( size_t ii = 0; ii < activeFilters_.size(); ++ii ) {
                if( !activeFilters_[ii] )
                    continue;
                const auto & activeSigmas = filtersToSigmas_[ii];
                auto & filter = filters_.at(ii);
                for( size_t jj = 0; jj < sigmas_.size(); ++jj ) {
                    if( !activeSigmas[jj] )
                        continue;
                    auto sigma = sigmas_[jj];
                    const auto & shapeView = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;
                    auto view = out.view(base.begin(), shapeView.begin()).squeezedView();
                    (*filter)(ff, view, sigma);
                    base[0] += filter->isMultiChannel() ? int64_t(DIM) : 1L;
                }
            }
        }
        
        inline void applyFiltersSequentialWithPresmoothing(const marray::View<float> & in,
                marray::View<float> & out,
                const Coord & shapeSingleChannel,
                const Coord & shapeMultiChannel,
                Coord & base ) const {
            
            // initialize for presmoothing 
            double sigmaPre = 0.;
            FastfiltersArrayType ff;
            detail_fastfilters::convertMarray2ff(in, ff);
            Coord preBase;
            for(int d = 0; d < DIM; ++d)
                preBase[d] = 0;
            marray::Marray<float> preSmoothed(shapeSingleChannel.begin(), shapeSingleChannel.end());
            
            // determine start coordinates to run with presmoothing
            std::vector<std::vector<Coord>> bases(sigmas_.size());
            for(size_t ii = 0; ii < bases.size(); ++ii)
                bases[ii] = std::vector<Coord>(filtersToSigmas_.size());
            int64_t channelStart = 0;
            for( size_t jj = 0; jj < filtersToSigmas_.size(); ++jj ) {
                for( size_t ii = 0; ii < sigmas_.size(); ++ii ) {
                    if( !filtersToSigmas_[jj][ii] )
                        continue;
                    Coord channelBase = base;
                    channelBase[0] = channelStart;
                    bases[ii][jj] = channelBase;
                    channelStart += filters_.at(jj)->isMultiChannel() ? int64_t(DIM) : 1L;
                }
            }

            // iterate over sigmas and apply filters with pre smoothing
            for(size_t ii = 0; ii < sigmas_.size(); ++ii) {
                // determine the correct sigma for pre smoothing
                double sigma = sigmas_[ii];
                //std::cout << "Sigma: " << ii << " = " << sigma << std::endl; 
                NIFTY_CHECK_OP(sigma,>,sigmaPre,"Presmoothing only works for ascending sigmas!");
                double sigmaNeed = std::sqrt(sigma*sigma - sigmaPre*sigmaPre); 
                if( sigmaNeed > 1. ) {
                    // determine the sigma we use for presmoothing
                    double sigmaPreDesired = std::sqrt(sigma*sigma - 1.);
                    double sigmaNeedForPre = std::sqrt(sigmaPreDesired*sigmaPreDesired - sigmaPre*sigmaPre);
                    // presmooth with gaussian (assume that we have gaussianSmoothing as filter, 
                    // otherwise this does not make sense...)
                    auto preView = preSmoothed.view( preBase.begin(), shapeSingleChannel.begin() ).squeezedView();
                    //std::cout << "Presmoothing with " << sigmaNeedForPre << std::endl;
                    (*(filters_.at(0)))(ff, preView, sigmaNeedForPre);
                    detail_fastfilters::convertMarray2ff(preView, ff); // write presmoothed into the ff array
                    sigmaPre = sigmaPreDesired;
                    sigmaNeed = std::sqrt(sigma*sigma - sigmaPre*sigmaPre);
                }

                // apply all filters for this sigma
                for(size_t jj = 0; jj < filtersToSigmas_.size(); ++jj) {
                    if( !filtersToSigmas_[jj][ii] )
                        continue;
                    //std::cout << "Filter " << jj << std::endl;
                    auto & filter = filters_.at(jj);
                    const auto & viewBase = bases[ii][jj]; 
                    const auto & viewShape = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;
                    auto view = out.view( viewBase.begin(), viewShape.begin() ).squeezedView();
                    //std::cout << "Applying with " << sigmaNeed << std::endl;
                    (*filter)(ff, view, sigmaNeed);
                }
            }
        }
          
        inline void applyFiltersParallel(const marray::View<float> & in,
                marray::View<float> & out,
                const Coord & shapeSingleChannel,
                const Coord & shapeMultiChannel,
                Coord & base,
                parallel::ThreadPool & threadpool ) const {
            
            // copy in-marray to fastfilters array
            FastfiltersArrayType ff;
            detail_fastfilters::convertMarray2ff(in, ff);
            
            // determine start coordinates to run in parallel
            std::vector<std::pair<size_t,double>> filterIdAndSigmas;
            std::vector<Coord> bases;
            int64_t channelStart = 0;
            for( size_t ii = 0; ii < activeFilters_.size(); ++ii ) {
                if( !activeFilters_[ii] )
                    continue;
                const auto & activeSigmas = filtersToSigmas_[ii];
                for( size_t jj = 0; jj < sigmas_.size(); ++jj ) {
                    if( !activeSigmas[jj] )
                        continue;
                    filterIdAndSigmas.push_back(std::make_pair(ii, sigmas_[jj]));
                    Coord channelBase = base;
                    channelBase[0] = channelStart;
                    bases.push_back(channelBase);
                    channelStart += filters_.at(ii)->isMultiChannel() ? int64_t(DIM) : 1L;
                }
            }
            
            // apply filters in parallel
            parallel::parallel_foreach(threadpool, filterIdAndSigmas.size(), [&](const int tid, const int64_t fid){
                auto & filter = filters_.at(filterIdAndSigmas[fid].first);
                const auto sigma = filterIdAndSigmas[fid].second;
                const auto & viewBase = bases[fid]; 
                const auto & viewShape = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;
                auto view = out.view( viewBase.begin(), viewShape.begin() ).squeezedView();
                (*filter)(ff, view, sigma);
            });
        }

        std::vector<double> sigmas_;
        FiltersToSigmasType filtersToSigmas_;
        std::map<size_t,FilterBase*> filters_;
        std::array<bool,4> activeFilters_;
    };
    
    

} // end namespace features
} // end namespace nifty


