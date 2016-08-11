#pragma once
#ifndef NIFTY_FEATURES_FASTFILTERS_WRAPPER_HXX
#define NIFTY_FEATURES_FASTFILTERS_WRAPPER_HXX

#include "nifty/marray/marray.hxx"
#include "fastfilters.h"

namespace nifty{
namespace features{

namespace detail_fastfilters {
    
    // copied from fastfilters/python/core.hxx
    template <typename fastfilters_array_t> struct ff_ndim_t {
    };
    
    template <> struct ff_ndim_t<fastfilters_array2d_t> {
        static const unsigned int ndim = 2;
    
        static void set_z(size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
        {
        }
        static void set_stride_z(size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
        {
        }
    };
    
    template <> struct ff_ndim_t<fastfilters_array3d_t> {
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
    void convert_marray2ff(const marray::View<float> & array, fastfilters_array_t & ff) {
        
        const unsigned int dim = ff_ndim_t<fastfilters_array_t>::ndim;
    
        if (array.dimension() >= (int) dim) {
            ff.ptr = (float *) &array(0);
    
            ff.n_x = array.shape(dim - 1);
            ff.stride_x = array.strides(dim - 1);
    
            ff.n_y = array.shape(dim - 2);
            ff.stride_y = array.strides(dim - 2);
    
            if (dim == 3) {
                ff_ndim_t<fastfilters_array_t>::set_z(array.shape(dim - 3), ff);
                ff_ndim_t<fastfilters_array_t>::set_stride_z(array.strides(dim - 3), ff);
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

    struct FilterBase {
        
        FilterBase() {
            if(!initCalled_) {
                fastfilters_init();
                initCalled_ = true;
            }
            opt_.window_ratio = 0.;
        }

        virtual void operator()(const fastfilters_array2d_t &, marray::View<float> &, const double) const = 0;

        virtual void operator()(const fastfilters_array3d_t &, marray::View<float> &, const  double) const = 0; 

        virtual bool isMultiChannel() const = 0;

        void set_window_ratio(const double ratio) {
            opt_.window_ratio = ratio;
        }

        static bool initCalled_;
    protected:
        fastfilters_options_t opt_;
    };

    bool FilterBase::initCalled_ = false;

    struct GaussianSmoothing : FilterBase {
        
        void operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convert_marray2ff(out, ff_out);
            if( !fastfilters_fir_gaussian2d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianSmoothing 2d failed.");
        }

        void operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convert_marray2ff(out, ff_out);
            if( !fastfilters_fir_gaussian3d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianSmoothing 3d failed.");
        }

        bool isMultiChannel() const {
            return false;
        }

    };
    
    struct LaplacianOfGaussian : FilterBase {
        
        void operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma) const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convert_marray2ff(out, ff_out);
            if( !fastfilters_fir_laplacian2d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("LaplacianOfGaussian 2d failed!");
        }

        void operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convert_marray2ff(out, ff_out);
            if( !fastfilters_fir_laplacian3d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("LaplacianOfGaussian 3d failed!");
        }
        
        bool isMultiChannel() const {
            return false;
        }

    };
    

    struct GaussianGradientMagnitude : FilterBase {
        
        void operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convert_marray2ff(out, ff_out);
            if( !fastfilters_fir_gradmag2d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianGradientMagnitude 2d failed!");
        }

        void operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma)  const {
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convert_marray2ff(out, ff_out);
            if( !fastfilters_fir_gradmag3d(&ff, sigma, &ff_out, &opt_) )
                throw std::logic_error("GaussianGradientMagnitude 3d failed!");
        }
        
        bool isMultiChannel() const {
            return false;
        }
        

    };
    
    struct HessianOfGaussianEigenvalues : FilterBase {
        
        void operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {
            
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

        void operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
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
        
        bool isMultiChannel() const {
            return true;
        }

    };
    

    // outer scale has to be set as member variable to keep consistency w/ the operator()
    struct StructureTensorEigenvalues : FilterBase {

        StructureTensorEigenvalues() : FilterBase(), sigmaOuter_(0.) {
        }
        
        void operator()(const fastfilters_array2d_t & ff, marray::View<float> & out, const  double sigma)  const {

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

        void operator()(const fastfilters_array3d_t & ff, marray::View<float> & out, const  double sigma) const {
            
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
        
        bool isMultiChannel() const {
            return true;
        }

        void setOuterScale(const double sigmaOuter) {
            sigmaOuter_ = sigmaOuter;
        }

    private:
        double sigmaOuter_;

    };


    // wrap fastfilters in a functor
    template<unsigned DIM> struct ApplyFilters;

    template<> struct ApplyFilters<2> {
        
        typedef fastfilters_array2d_t FastfiltersArrayType;
        
        ApplyFilters(const std::vector<double> & sigmas, const std::vector<FilterBase*> & filters) : sigmas_(sigmas), filters_(filters) {
        }
        
        void operator()(const marray::View<float> & in, marray::View<float> & out) const{
    
            NIFTY_CHECK_OP(in.dimension(),==,2,"Input needs to be 2 dimensional.")
            NIFTY_CHECK_OP(out.shape(0),==,numberOfChannels(),"Number of Channels of out Array do not match!")
            NIFTY_CHECK_OP(out.shape(1),==,in.shape(0),"y-axis does not agree")
            NIFTY_CHECK_OP(out.shape(2),==,in.shape(1),"x-axis does not agree")

            const size_t shapeSingleChannel[] = {1, out.shape(0), out.shape(1)};
            const size_t shapeMultiChannel[]  = {2, out.shape(0), out.shape(1)};
            
            FastfiltersArrayType ff;
            detail_fastfilters::convert_marray2ff(in, ff);

            size_t base[] = {0,0,0};
            
            for( auto filter : filters_ ) {
                size_t nChannels = filter->isMultiChannel() ? 2 : 1;
                const auto & shapeView = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;
                for( auto sigma : sigmas_) {
                    auto view = out.view(base, shapeView);
                    view.squeeze();
                    (*filter)(ff, view, sigma);
                    base[0] += nChannels;
                }
            }
            
        }

        size_t numberOfChannels() const {
            size_t numberOfChannels = 0;
            for(auto filter : filters_)
                numberOfChannels += filter->isMultiChannel() ? 2 : 1;
            return numberOfChannels * sigmas_.size();
        }
        
    private:
        std::vector<double> sigmas_;
        std::vector<FilterBase*> filters_;
    };
    

    template<> struct ApplyFilters<3> {
        
        typedef fastfilters_array3d_t FastfiltersArrayType;
        
        ApplyFilters(const std::vector<double> & sigmas, const std::vector<FilterBase*> filters) : sigmas_(sigmas), filters_(filters) {
            fastfilters_init(); // FIXME this might cause problems if we init more than one ApplyFilters
        }
        
        void operator()(const marray::View<float> & in, marray::View<float> & out) const{
    
            NIFTY_CHECK_OP(in.dimension(),==,3,"Input needs to be 3 dimensional.")
            NIFTY_CHECK_OP(out.shape(0),==,numberOfChannels(),"Number of Channels of out Array do not match!")
            NIFTY_CHECK_OP(out.shape(1),==,in.shape(0),"z-axis does not agree")
            NIFTY_CHECK_OP(out.shape(2),==,in.shape(1),"y-axis does not agree")
            NIFTY_CHECK_OP(out.shape(3),==,in.shape(2),"x-axis does not agree")

            const size_t shapeSingleChannel[] = {1, out.shape(0), out.shape(1), out.shape(3)};
            const size_t shapeMultiChannel[]  = {3, out.shape(0), out.shape(1), out.shape(3)};
            
            FastfiltersArrayType ff;
            detail_fastfilters::convert_marray2ff(in, ff);

            size_t base[] = {0,0,0,0};
            
            for( auto filter : filters_ ) {
                for( auto sigma : sigmas_) {
                    size_t nChannels = filter->isMultiChannel() ? 3 : 1;
                    const auto & shapeView = filter->isMultiChannel() ? shapeMultiChannel : shapeSingleChannel;
                    auto view = out.view(base, shapeView);
                    view.squeeze();
                    (*filter)(ff, view, sigma);
                    base[0] += nChannels;
                }
            }
        }

        size_t numberOfChannels() const {
            size_t numberOfChannels = 0;
            for(auto filter : filters_)
                numberOfChannels += filter->isMultiChannel() ? 3 : 1;
            return numberOfChannels * sigmas_.size();
        }
            
        
    private:
        std::vector<double> sigmas_;
        std::vector<FilterBase*> filters_;
    };
    
    

} // end namespace features
} // end namespace nifty


#endif /* NIFTY_FEATURES_FASTFILTERS_WRAPPER_HXX */
