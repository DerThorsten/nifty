#pragma once

#include <mutex>
#include <map>
#include <cmath>

#include "fastfilters.h"

#include "nifty/parallel/threadpool.hxx"
#include "nifty/array/arithmetic_array.hxx"

#include "nifty/xtensor/xtensor.hxx"

namespace nifty{
namespace features{

namespace detail_fastfilters {

    // copied from fastfilters/python/core.hxx
    template <typename fastfilters_array_t> struct FastfiltersDim {
    };

    template <> struct FastfiltersDim<fastfilters_array2d_t> {
        static const unsigned int ndim = 2;

        static void set_z(std::size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
        {
        }
        static void set_stride_z(std::size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
        {
        }
    };

    template <> struct FastfiltersDim<fastfilters_array3d_t> {
        static const unsigned int ndim = 3;

        static void set_z(std::size_t n_z, fastfilters_array3d_t &k)
        {
            k.n_z = n_z;
        }
        static void set_stride_z(std::size_t stride_z, fastfilters_array3d_t &k)
        {
            k.stride_z = stride_z;
        }
    };


    // adapted from fastfilters/python/core.hxx, convert_py2ff
    template <typename fastfilters_array_t, typename ARRAY>
    void convertXtensor2ff(const xt::xexpression<ARRAY> & arrayExp, fastfilters_array_t & ff) {

        const auto & array = arrayExp.derived_cast();
        const unsigned int dim = FastfiltersDim<fastfilters_array_t>::ndim;
        const auto & shape = array.shape();
        const auto & strides = array.strides();

        if (array.dimension() >= (int) dim) {
            ff.ptr = (float *) &array(0);

            ff.n_x = shape[dim - 1];
            ff.stride_x = strides[dim - 1];

            ff.n_y = shape[dim - 2];
            ff.stride_y = strides[dim - 2];

            if (dim == 3) {
                FastfiltersDim<fastfilters_array_t>::set_z(shape[dim - 3], ff);
                FastfiltersDim<fastfilters_array_t>::set_stride_z(strides[dim - 3], ff);
            }

            //ff.n_x = shape[0];
            //ff.stride_x = strides[0];

            //ff.n_y = shape[0];
            //ff.stride_y = strides[0];

            //if (dim == 3) {
            //    FastfiltersDim<fastfilters_array_t>::set_z(shape[0], ff);
            //    FastfiltersDim<fastfilters_array_t>::set_stride_z(strides[0], ff);
            //}
        } else {
            throw std::runtime_error("Too few dimensions.");
        }

        if (array.dimension() == dim) {
            ff.n_channels = 1;
        }
        else if ((array.dimension() == dim + 1) && shape[dim] < 8 && strides[dim] == 1) {
            ff.n_channels = shape[dim];
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

        void set_window_ratio(const double ratio) {
            opt_.window_ratio = ratio;
        }

    protected:
        fastfilters_options_t opt_;
    };


    struct GaussianSmoothing : FilterBase {

        template<class ARRAY>
        void inline operator()(const fastfilters_array2d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertXtensor2ff(out, ff_out);
            if(!fastfilters_fir_gaussian2d(&ff, 0, sigma, &ff_out, &opt_)) {
                throw std::runtime_error("GaussianSmoothing 2d failed.");
            }
        }

        template<class ARRAY>
        void inline operator()(const fastfilters_array3d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertXtensor2ff(out, ff_out);
            if( !fastfilters_fir_gaussian3d(&ff, 0, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianSmoothing 3d failed.");
        }

        bool isMultiChannel() const {
            return false;
        }

    };


    struct LaplacianOfGaussian : FilterBase {

        template<class ARRAY>
        void inline operator()(const fastfilters_array2d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertXtensor2ff(out, ff_out);
            if( !fastfilters_fir_laplacian2d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("LaplacianOfGaussian 2d failed!");
        }

        template<class ARRAY>
        void inline operator()(const fastfilters_array3d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const  double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertXtensor2ff(out, ff_out);
            if( !fastfilters_fir_laplacian3d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("LaplacianOfGaussian 3d failed!");
        }

        bool inline isMultiChannel() const {
            return false;
        }

    };


    struct GaussianGradientMagnitude : FilterBase {

        template<class ARRAY>
        void inline operator()(const fastfilters_array2d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array2d_t ff_out;
            detail_fastfilters::convertXtensor2ff(out, ff_out);
            if( !fastfilters_fir_gradmag2d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianGradientMagnitude 2d failed!");
        }

        template<class ARRAY>
        void inline operator()(const fastfilters_array3d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array3d_t ff_out;
            detail_fastfilters::convertXtensor2ff(out, ff_out);
            if( !fastfilters_fir_gradmag3d(&ff, sigma, &ff_out, &opt_) )
                throw std::runtime_error("GaussianGradientMagnitude 3d failed!");
        }

        bool inline isMultiChannel() const {
            return false;
        }
    };


    struct HessianOfGaussianEigenvalues : FilterBase {

        template<class ARRAY>
        void inline operator()(const fastfilters_array2d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {

            auto & out = outExp.derived_cast();
            fastfilters_array2d_t * xx = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * yy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * xy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);

            if( !fastfilters_fir_hog2d(&ff, sigma, xx, xy, yy, &opt_) ) {
                throw std::runtime_error("HessianOfGaussian 2d failed.");
            }

            const std::size_t numberOfPixels = ff.n_x * ff.n_y;

            float* ev0 = &out(0);
            float* ev1 = &out(0) + numberOfPixels;

            fastfilters_linalg_ev2d(xx->ptr, xy->ptr, yy->ptr, ev0, ev1, numberOfPixels);

            fastfilters_array2d_free(xx);
            fastfilters_array2d_free(yy);
            fastfilters_array2d_free(xy);
        }

        template<class ARRAY>
        void inline operator()(const fastfilters_array3d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {
            auto & out = outExp.derived_cast();
            fastfilters_array3d_t * xx = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * zz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);

            if( !fastfilters_fir_hog3d(&ff, sigma, xx, yy, zz, xy, xz, yz, &opt_) ) 
                throw std::runtime_error("HessianOfGaussian 3d failed.");

            const std::size_t numberOfPixels = ff.n_x * ff.n_y * ff.n_z;

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

        template<class ARRAY>
        void inline operator()(const fastfilters_array2d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {

            NIFTY_CHECK_OP(sigma, <, sigmaOuter_,
                           "inner scale has to be smaller than outer scale, set via setOuterScale")

            auto & out = outExp.derived_cast();
            fastfilters_array2d_t * xx = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * yy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);
            fastfilters_array2d_t * xy = fastfilters_array2d_alloc(ff.n_x, ff.n_y, 1);

            if( !fastfilters_fir_structure_tensor2d(&ff, sigma, sigmaOuter_, xx, xy, yy, &opt_) ) 
                throw std::runtime_error("StructurTensor 2d failed.");

            const std::size_t numberOfPixels = ff.n_x * ff.n_y;

            float* ev0 = &out(0);
            float* ev1 = &out(0) + numberOfPixels;

            fastfilters_linalg_ev2d(xx->ptr, xy->ptr, yy->ptr, ev0, ev1, numberOfPixels);

            fastfilters_array2d_free(xx);
            fastfilters_array2d_free(yy);
            fastfilters_array2d_free(xy);

        }

        template<class ARRAY>
        void inline operator()(const fastfilters_array3d_t & ff,
                               xt::xexpression<ARRAY> & outExp,
                               const double sigma) const {

            NIFTY_CHECK_OP(sigma, <, sigmaOuter_,
                           "inner scale has to be smaller than outer scale, set via setOuterScale")

            auto & out = outExp.derived_cast();
            fastfilters_array3d_t * xx = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * zz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xy = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * xz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);
            fastfilters_array3d_t * yz = fastfilters_array3d_alloc(ff.n_x, ff.n_y, ff.n_z, 1);

            if( !fastfilters_fir_structure_tensor3d(&ff, sigma, 2*sigma, xx, yy, zz, xy, xz, yz, &opt_) )
                throw std::runtime_error("StructureTensor 3d failed.");

            const std::size_t numberOfPixels = ff.n_x * ff.n_y * ff.n_z;

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

        // TODO no structure tensor for now due to second scale
        // enum encoding the different filters
        enum filterNames {GaussianSmoothing = 0,
                          LaplacianOfGaussian = 1,
                          GaussianGradientMagnitude = 2,
                          HessianOfGaussianEigenvalues = 3};

        typedef std::array<std::vector<bool>, 4> FiltersToSigmasType;

        ApplyFilters(const std::vector<double> & sigmas,
                     const FiltersToSigmasType & filtersToSigmas) : sigmas_(sigmas),
                                                                    filtersToSigmas_(filtersToSigmas){

            // initialize the filters according to filtersToSigmas
            for(const auto & filtToSig : filtersToSigmas) {
                NIFTY_CHECK_OP(filtToSig.size(), ==, sigmas_.size(), "");
            }

            for(std::size_t ii = 0; ii < filtersToSigmas_.size(); ++ii) {
                // if at least one sigma is active, we push back the corresponding filter
                const auto & filtToSig = filtersToSigmas_[ii];
                activeFilters_[ii] = std::any_of(filtToSig.begin(), filtToSig.end(), [](bool val){return val;});
            }
        }

        // apply filters sequential
        template<class ARRAY1, class ARRAY2>
        void operator()(const xt::xexpression<ARRAY1> & in,
                        xt::xexpression<ARRAY2> & out,
                        const bool presmooth = false) const {
            Coord shapeSingleChannel, shapeMultiChannel, base;
            applyCommon(in, out, shapeSingleChannel, shapeMultiChannel, base);
            if(presmooth) {
                applyFiltersSequentialWithPresmoothing(in, out, shapeSingleChannel, shapeMultiChannel, base);
            } else {
                applyFiltersSequential(in, out, shapeSingleChannel, shapeMultiChannel, base);
            }
        }

        // apply filters in parallel
        template<class ARRAY1, class ARRAY2>
        void operator()(const xt::xexpression<ARRAY1> & in,
                        xt::xexpression<ARRAY2> & out,
                        parallel::ThreadPool & threadpool) const{
            Coord shapeSingleChannel, shapeMultiChannel, base;
            applyCommon(in, out, shapeSingleChannel, shapeMultiChannel, base);
            applyFiltersParallel(in, out, shapeSingleChannel, shapeMultiChannel, base, threadpool);
        }

        std::size_t numberOfChannels() const {
            std::size_t numberOfChannels = 0;
            for(std::size_t ii = 0; ii < activeFilters_.size(); ++ii) {
                if(!activeFilters_[ii]) {
                    continue;
                }
                std::size_t nChans = this->numberOfChannels(ii);
                const auto & activeSigmas = filtersToSigmas_[ii];
                for( std::size_t jj = 0; jj < sigmas_.size(); ++jj ) {
                    if( activeSigmas[jj] )
                        numberOfChannels += nChans;
                }
            }
            return numberOfChannels;
        }

    private:

        inline std::size_t numberOfChannels(const std::size_t filterId) const {
            // only the hessian of gaussian eigenvalues (filter id = 3)
            // are multichannel
            return filterId == 3 ? DIM : 1;
        }

        template<class ARRAY>
        inline void applyFilterId(const std::size_t filterId,
                                  FastfiltersArrayType & ff,
                                  xt::xexpression<ARRAY> & outExp,
                                  const double sigma) const {
            auto & out = outExp.derived_cast();
            switch(filterId) {
                case 0: gs_(ff, out, sigma); break;
                case 1: log_(ff, out, sigma); break;
                case 2: ggm_(ff, out, sigma); break;
                case 3: hog_(ff, out, sigma); break;
                default: throw std::runtime_error("Impossible!");
            }
        }

        // common pre-processing for all overloads of operator()
        template<class ARRAY1, class ARRAY2>
        inline void applyCommon(const xt::xexpression<ARRAY1> & inExp,
                                xt::xexpression<ARRAY2> & outExp,
                                Coord & shapeSingleChannel,
                                Coord & shapeMultiChannel,
                                Coord & base) const{
            const auto & in = inExp.derived_cast();
            auto & out = outExp.derived_cast();
            // checks
            NIFTY_CHECK_OP(in.dimension(), ==, DIM,
                           "Input needs to be of correct dimension.");
            NIFTY_CHECK_OP(out.shape()[0], ==, numberOfChannels(),
                           "Number of channels of out array do not match!");
            const auto & shape = in.shape();
            for(int d = 0; d < DIM; ++d) {
                NIFTY_CHECK_OP(out.shape()[d + 1], ==, shape[d], "In and out axis do not agree!");
            }

            // determine shapes
            shapeSingleChannel[0] = 1L;
            shapeMultiChannel[0] = int64_t(DIM);
            base[0] = 0L;
            for(int d = 0; d < DIM; d++) {
               shapeSingleChannel[d+1] = shape[d];
               shapeMultiChannel[d+1] = shape[d];
               base[d+1] = 0L;
            }
        }

        template<class ARRAY1, class ARRAY2>
        inline void applyFiltersSequential(const xt::xexpression<ARRAY1> & inExp,
                                           xt::xexpression<ARRAY2> & outExp,
                                           const Coord & shapeSingleChannel,
                                           const Coord & shapeMultiChannel,
                                           Coord & base) const {
            const auto & in = inExp.derived_cast();
            auto & out = outExp.derived_cast();

            // copy in-xtensor to fastfilters array
            FastfiltersArrayType ff;
            detail_fastfilters::convertXtensor2ff(in, ff);

            // apply filters sequentially
            for( std::size_t ii = 0; ii < activeFilters_.size(); ++ii ) {
                if(!activeFilters_[ii]) {
                    continue;
                }
                const auto & activeSigmas = filtersToSigmas_[ii];
                for(std::size_t jj = 0; jj < sigmas_.size(); ++jj) {
                    if(!activeSigmas[jj]) {
                        continue;
                    }
                    auto sigma = sigmas_[jj];
                    const auto & shapeView = (numberOfChannels(ii) == 1)  ? shapeSingleChannel : shapeMultiChannel;;

                    xt::slice_vector slice;
                    xtensor::sliceFromOffset(slice, base, shapeView);
                    auto view = xt::strided_view(out, slice);
                    auto squeezedView = xtensor::squeezedView(view);
                    applyFilterId(ii, ff, squeezedView, sigma);
                    base[0] += (int64_t) numberOfChannels(ii);
                }
            }
        }

        template<class ARRAY1, class ARRAY2>
        inline void applyFiltersSequentialWithPresmoothing(const xt::xexpression<ARRAY1> & inExp,
                                                           xt::xexpression<ARRAY2> & outExp,
                                                           const Coord & shapeSingleChannel,
                                                           const Coord & shapeMultiChannel,
                                                           Coord & base) const {
            typedef typename xt::xtensor<float, DIM + 1>::shape_type ShapeType;
            const auto & in = inExp.derived_cast();
            auto & out = outExp.derived_cast();

            // initialize for presmoothing
            double sigmaPre = 0.;
            FastfiltersArrayType ff;
            detail_fastfilters::convertXtensor2ff(in, ff);
            Coord preBase;
            for(int d = 0; d < DIM; ++d) {
                preBase[d] = 0;
            }

            ShapeType arrayShape;
            std::copy(shapeSingleChannel.begin(), shapeSingleChannel.end(), arrayShape.begin());
            xt::xtensor<float, DIM + 1> preSmoothed(arrayShape);

            // determine start coordinates to run with presmoothing
            std::vector<std::vector<Coord>> bases(sigmas_.size());
            for(std::size_t ii = 0; ii < bases.size(); ++ii) {
                bases[ii] = std::vector<Coord>(filtersToSigmas_.size());
            }

            int64_t channelStart = 0;
            for( std::size_t jj = 0; jj < filtersToSigmas_.size(); ++jj ) {
                for( std::size_t ii = 0; ii < sigmas_.size(); ++ii ) {
                    if(!filtersToSigmas_[jj][ii]) {
                        continue;
                    }
                    Coord channelBase = base;
                    channelBase[0] = channelStart;
                    bases[ii][jj] = channelBase;
                    channelStart += numberOfChannels(jj);
                }
            }

            // iterate over sigmas and apply filters with pre smoothing
            for(std::size_t ii = 0; ii < sigmas_.size(); ++ii) {

                // determine the correct sigma for pre smoothing
                double sigma = sigmas_[ii];
                NIFTY_CHECK_OP(sigma, >, sigmaPre,
                               "Presmoothing only works for ascending sigmas!");
                double sigmaNeed = std::sqrt(sigma*sigma - sigmaPre*sigmaPre);
                if( sigmaNeed > 1. ) {
                    // determine the sigma we use for presmoothing
                    double sigmaPreDesired = std::sqrt(sigma*sigma - 1.);
                    double sigmaNeedForPre = std::sqrt(sigmaPreDesired*sigmaPreDesired - sigmaPre*sigmaPre);
                    // presmooth with gaussian
                    xt::slice_vector slice;
                    xtensor::sliceFromOffset(slice, preBase, shapeSingleChannel);
                    auto preView = xt::strided_view(preSmoothed, slice);
                    auto squeezedPreView = xtensor::squeezedView(preView);
                    gs_(ff, squeezedPreView, sigmaNeedForPre);
                    // write presmoothed into the ff array
                    detail_fastfilters::convertXtensor2ff(squeezedPreView, ff);
                    sigmaPre = sigmaPreDesired;
                    sigmaNeed = std::sqrt(sigma*sigma - sigmaPre*sigmaPre);
                }

                // apply all filters for this sigma
                for(std::size_t jj = 0; jj < filtersToSigmas_.size(); ++jj) {
                    if(!filtersToSigmas_[jj][ii]) {
                        continue;
                    }
                    const auto & viewBase = bases[ii][jj];
                    const auto & viewShape = (numberOfChannels(jj) == 1) ? shapeSingleChannel : shapeMultiChannel;

                    xt::slice_vector slice;
                    xtensor::sliceFromOffset(slice, viewBase, viewShape);
                    auto view = xt::strided_view(out, slice);
                    auto squeezedView = xtensor::squeezedView(view);
                    applyFilterId(jj, ff, squeezedView, sigmaNeed);
                }
            }
        }

        template<class ARRAY1, class ARRAY2>
        inline void applyFiltersParallel(const xt::xexpression<ARRAY1> & inExp,
                                         xt::xexpression<ARRAY2> & outExp,
                                         const Coord & shapeSingleChannel,
                                         const Coord & shapeMultiChannel,
                                         Coord & base,
                                         parallel::ThreadPool & threadpool) const {
            const auto & in = inExp.derived_cast();
            auto & out = outExp.derived_cast();

            // copy in-xtensor to fastfilters array
            FastfiltersArrayType ff;
            detail_fastfilters::convertXtensor2ff(in, ff);

            // determine start coordinates to run in parallel
            std::vector<std::pair<std::size_t, double>> filterIdAndSigmas;
            std::vector<Coord> bases;
            int64_t channelStart = 0;
            for(std::size_t ii = 0; ii < activeFilters_.size(); ++ii) {
                if(!activeFilters_[ii]) {
                    continue;
                }
                const auto & activeSigmas = filtersToSigmas_[ii];
                for(std::size_t jj = 0; jj < sigmas_.size(); ++jj) {
                    if(!activeSigmas[jj]) {
                        continue;
                    }
                    filterIdAndSigmas.push_back(std::make_pair(ii, sigmas_[jj]));
                    Coord channelBase = base;
                    channelBase[0] = channelStart;
                    bases.push_back(channelBase);
                    channelStart += (int64_t) numberOfChannels(ii);
                }
            }

            // apply filters in parallel
            parallel::parallel_foreach(threadpool, filterIdAndSigmas.size(), [&](const int tid, const int64_t fid){
                const auto filterId = filterIdAndSigmas[fid].first;
                const auto sigma = filterIdAndSigmas[fid].second;
                const auto & viewBase = bases[fid];
                const auto & viewShape = (numberOfChannels(filterId) == 1)  ? shapeSingleChannel : shapeMultiChannel;
                //std::cout << "Apply Filter from " << viewBase << " with shape " << viewShape << std::endl;
                xt::slice_vector slice;
                xtensor::sliceFromOffset(slice, viewBase, viewShape);
                auto view = xt::strided_view(out, slice);
                auto squeezedView = xtensor::squeezedView(view);
                applyFilterId(filterId, ff, squeezedView, sigma);
            });
        }

        std::vector<double> sigmas_;
        FiltersToSigmasType filtersToSigmas_;
        std::array<bool, 4> activeFilters_;

        // the individual filters
        // gaussian smoothing: filter id 0
        nifty::features::GaussianSmoothing gs_;
        // laplacian of gaussian: filter id 1
        nifty::features::LaplacianOfGaussian log_;
        // gaussian gradient magnidutde: filter id 2
        nifty::features::GaussianGradientMagnitude ggm_;
        // hesseian of gaussian eigenvalues: filter id 3
        nifty::features::HessianOfGaussianEigenvalues hog_;
    };


} // end namespace features
} // end namespace nifty


