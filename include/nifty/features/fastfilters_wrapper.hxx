#pragma once
#ifndef NIFTY_FEATURES_FASTFILTERS_WRAPPER_HXX
#define NIFTY_FEATURES_FASTFILTERS_WRAPPER_HXX

#include <mutex>

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
            throw std::logic_error("Too few dimensions.");
        }
    
        if (array.dimension() == dim) {
            ff.n_channels = 1;
        } 
        //else if ((array.dimension() == dim + 1) && array.shape(dim) < 8 && array.strides(dim) == sizeof(float)) {
        else if ((array.dimension() == dim + 1) && array.shape(dim) < 8 && array.strides(dim) == 1) {
            ff.n_channels = array.shape(dim);
        } else {
            throw std::logic_error("Invalid number of dimensions or too many channels or stride between channels.");
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
                std::cout << "Fastfilters initialized" << std::endl;
            });
            opt_.window_ratio = 0.; // TODO zero seems not to be a sensible default, as this means no border treatment. According to sven, there are some defaults in vigra, check that!
        }

        virtual ~FilterBase(){};

        virtual void inline operator()(const fastfilters_array2d_t &, marray::View<float> &, const double) const = 0;

        virtual void inline operator()(const fastfilters_array3d_t &, marray::View<float> &, const  double) const = 0; 

        virtual bool inline isMultiChannel() const = 0;
        
        virtual void setOuterScale(const double sigmaOuter) = 0;

        void set_window_ratio(const double ratio) {
            opt_.window_ratio = ratio; // maybe setting it here for the use case on hand makes more sense (modulu overloading for the actual filter)
        }

    protected:
        fastfilters_options_t opt_;
    };

    struct GaussianSmoothing : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gaussian2d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianSmoothing 2d failed.");
        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gaussian3d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianSmoothing 3d failed.");
        }

        bool isMultiChannel() const {
            return false;
        }
        
        void setOuterScale(const double sigmaOuter)
        {throw std::runtime_error("setOuterScale is not defined for GaussianSmoothing");}

    };
    
    struct LaplacianOfGaussian : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_laplacian2d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("LaplacianOfGaussian 2d failed!");
        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_laplacian3d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("LaplacianOfGaussian 3d failed!");
        }
        
        bool inline isMultiChannel() const {
            return false;
        }
        
        void setOuterScale(const double sigmaOuter)
        {throw std::runtime_error("setOuterScale is not defined for LaplacianOfGaussian");}

    };
    

    struct GaussianGradientMagnitude : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gradmag2d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianGradientMagnitude 2d failed!");
        }

        void inline operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertMarray2ff(out, ff_out);
            if( !fastfilters_fir_gradmag3d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianGradientMagnitude 3d failed!");
        }
        
        bool inline isMultiChannel() const {
            return false;
        }
        
        void setOuterScale(const double sigmaOuter)
        {throw std::runtime_error("setOuterScale is not defined for GaussianGradientMagnitude");}

    };
    
    struct HessianOfGaussianEigenvalues : FilterBase {
        
        void inline operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {
            
            fastfilters_array2d_t * xx = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * yy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * xy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);

            if( !fastfilters_fir_hog2d(&ff, sigma, xx, xy, yy, &opt_) ) 
                throw std::logic_error("HessianOfGaussian 2d failed.");

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
                throw std::logic_error("HessianOfGaussian 3d failed.");

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
        
        void setOuterScale(const double sigmaOuter)
        {throw std::runtime_error("setOuterScale is not defined for HessianOfGaussianEigenvalues");}

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
                throw std::logic_error("StructurTensor 2d failed.");

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
                throw std::logic_error("HessianOfGaussian 3d failed.");

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
        
        // construct from given sigmas and filter names
        ApplyFilters(const std::vector<double> & sigmas,
                const std::vector<std::string> & filterNames,
                const double outerScale = 0.) : sigmas_(sigmas), filters_() {
            
            fastfilters_init(); // FIXME this might cause problems if we init more than one ApplyFilters
                
            // init the vector with filter_type pointers
            for(const auto & filtName : filterNames) {
                if(filtName == "GaussianSmoothing")
                    filters_.emplace_back(new GaussianSmoothing());
                else if(filtName == "LaplacianOfGaussian")
                    filters_.emplace_back(new LaplacianOfGaussian());
                else if(filtName == "GaussianGradientMagnitude")
                    filters_.emplace_back(new GaussianGradientMagnitude());
                else if(filtName == "HessianOfGaussianEigenvalues")
                    filters_.emplace_back(new HessianOfGaussianEigenvalues());
                else if(filtName == "StructureTensorEigenvalues") {
                    filters_.emplace_back(new StructureTensorEigenvalues()); // TODO we don't use structure tensor for now, but we leave it in as an option
                    filters_.back()->setOuterScale(outerScale); // TODO check that this is non-zero, but maybe rethink for different outer scales
                }
                else
                    throw std::runtime_error("Unknown filter type!");
            }
        }
            
        // we need to make sure to delete the filter pointers TODO recheck this
        ~ApplyFilters() {
            //std::for_each(filters_.begin(), filters_.end(), []);
            for(auto & filter : filters_ )
                delete filter;
        }

        // apply filters sequential
        void operator()(const marray::View<float> & in, marray::View<float> & out) const{
    
            NIFTY_CHECK_OP(in.dimension(),==,DIM,"Input needs to be of correct dimension.")
            NIFTY_CHECK_OP(out.shape(0),==,numberOfChannels(),"Number of Channels of out Array do not match!")
            for(int d = 0; d < DIM; ++d){
                NIFTY_CHECK_OP(out.shape(d+1),==,in.shape(d),"in and out axis do not agree")
            }

            Coord shapeSingleChannel;
            Coord shapeMultiChannel;
            Coord base;
            
            shapeSingleChannel[0] = 1L;
            shapeMultiChannel[0] = int64_t(DIM);
            base[0] = 0L;
            for(int d = 0; d < DIM; d++) {
               shapeSingleChannel[d+1] = in.shape(d); 
               shapeMultiChannel[d+1] = in.shape(d); 
               base[d+1] = 0L;
            }
            
            FastfiltersArrayType ff;
            detail_fastfilters::convertMarray2ff(in, ff);

            for( auto filter : filters_ ) {
                for( auto sigma : sigmas_) {
                    
                    const auto & shapeView = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;
                    auto view = out.view(base.begin(), shapeView.begin()).squeezedView();
                    (*filter)(ff, view, sigma);
                    base[0] += filter->isMultiChannel() ? int64_t(DIM) : 1L;
                }
            }
        }
        
        // apply filters in parallel
        // TODO use tbb threadpool!
        void operator()(const marray::View<float> & in, marray::View<float> & out, parallel::ThreadPool & threadpool) const{
    
            NIFTY_CHECK_OP(in.dimension(),==,DIM,"Input needs to be of correct dimension.")
            NIFTY_CHECK_OP(out.shape(0),==,numberOfChannels(),"Number of Channels of out Array do not match!")
            for(int d = 0; d < DIM; ++d){
                NIFTY_CHECK_OP(out.shape(d+1),==,in.shape(d),"in and out axis do not agree")
            }

            Coord shapeSingleChannel;
            Coord shapeMultiChannel;
            Coord base;
            
            shapeSingleChannel[0] = 1L;
            shapeMultiChannel[0] = int64_t(DIM);
            base[0] = 0L;
            for(int d = 0; d < DIM; d++) {
               shapeSingleChannel[d+1] = in.shape(d); 
               shapeMultiChannel[d+1] = in.shape(d); 
               base[d+1] = 0L;
            }

            FastfiltersArrayType ff;
            detail_fastfilters::convertMarray2ff(in, ff);
            
            std::vector<std::pair<int,double>> filterIdAndSigmas;
            std::vector<Coord> bases;
            int64_t channelStart = 0;
            for( int filterId = 0; filterId < filters_.size(); ++filterId  ) {
                for( auto sigma : sigmas_ ) {
                    filterIdAndSigmas.push_back(std::make_pair(filterId, sigma));
                    Coord channelBase = base;
                    channelBase[0] = channelStart;
                    bases.push_back(channelBase);
                    channelStart += filters_[filterId]->isMultiChannel() ? int64_t(DIM) : 1L;
                }
            }

            parallel::parallel_foreach(threadpool, filterIdAndSigmas.size(), [&](const int tid, const int64_t fid){
                
                auto filter = filters_[filterIdAndSigmas[fid].first];
                const auto sigma = filterIdAndSigmas[fid].second;
                const auto & viewBase = bases[fid]; 
                const auto & viewShape = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;

                auto view = out.view( viewBase.begin(), viewShape.begin() ).squeezedView();
                (*filter)(ff, view, sigma);
            });
        }

        size_t numberOfChannels() const {
            size_t numberOfChannels = 0;
            for(auto filter : filters_)
                numberOfChannels += filter->isMultiChannel() ? DIM : 1;
            return numberOfChannels * sigmas_.size();
        }
            
        
    private:
        std::vector<double> sigmas_;
        std::vector<FilterBase*> filters_;
    };
    
    

} // end namespace features
} // end namespace nifty


#endif /* NIFTY_FEATURES_FASTFILTERS_WRAPPER_HXX */
