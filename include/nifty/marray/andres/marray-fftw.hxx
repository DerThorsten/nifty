#pragma once
#ifndef ANDRES_MARRAY_FFTW_HXX
#define ANDRES_MARRAY_FFTW_HXX

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <complex>

#include "marray.hxx"

#include <fftw3.h>

namespace andres {

template<class S = std::size_t>
class FFT {
public:
    typedef S size_type;

    FFT(const andres::Marray<double>&, andres::Marray<std::complex<double> >&);
    ~FFT();
    void execute();

private:
    fftw_plan plan_;
};

template<class S = std::size_t>
class IFFT {
public:
    typedef S size_type;

    IFFT(const andres::Marray<std::complex<double> >& in, andres::Marray<double>& out);
    ~IFFT();
    void execute();

private:
    fftw_plan plan_;
};

template<class S>
inline
FFT<S>::FFT(
    const andres::Marray<double>& in,
    andres::Marray<std::complex<double> >& out
) {
    if(in.coordinateOrder() != andres::CoordinateOrder::FirstMajorOrder) {
        throw std::runtime_error("Marray not in first-major order");
    }
    if(out.coordinateOrder() != andres::CoordinateOrder::FirstMajorOrder) {
        throw std::runtime_error("Marray not in first-major order");
    }
    if(in.dimension() != 2) {
        throw std::runtime_error("Marray not 2-dimensional.");
    }

    const int nRows = in.shape(0);
    const int nCols = in.shape(1);
    {
        int shape[] = {nRows, nCols / 2 + 1};
        out.resize(shape, shape + 2);
    }
    const int rank = 1; // 1D transform
    int n[] = {nCols}; // 1D transforms of length nCols
    int howmany = nRows; // nRows 1D transforms
    int idist = nCols; // number of columns in input array
    int odist = nCols/2+1; // number of columns in output array
    int istride = 1; int ostride = 1; // array is contiguous in memory
    int* inembed = n; int *onembed = n;
#   pragma omp critical
    {
        plan_ = fftw_plan_many_dft_r2c(
            rank, n, howmany,
            &in(0), inembed, istride, idist,
            reinterpret_cast<fftw_complex*>(&out(0)), onembed, ostride, odist,
            FFTW_ESTIMATE
        );
    }
}

template<class S>
inline
FFT<S>::~FFT() {
#   pragma omp critical
    fftw_destroy_plan(plan_);
}

template<class S>
inline void
FFT<S>::execute() {
    fftw_execute(plan_);
}

template<class S>
IFFT<S>::IFFT(
    const andres::Marray<std::complex<double> >& in,
    andres::Marray<double>& out
) {
    if(in.coordinateOrder() != andres::CoordinateOrder::FirstMajorOrder) {
        throw std::runtime_error("Marray not in first-major order");
    }
    if(out.coordinateOrder() != andres::CoordinateOrder::FirstMajorOrder) {
        throw std::runtime_error("Marray not in first-major order");
    }
    if(in.dimension() != 2) {
        throw std::runtime_error("Marray not 2-dimensional.");
    }

    const int nRows = in.shape(0);
    const int nCols = in.shape(1);
    {
        int shape[] = {nRows, nCols};
        out.resize(shape, shape + 2);
    }
    const int rank = 1; // 1D transform
    int n[] = {nCols}; // 1D transforms of length nCols
    int howmany = nRows; // nRows 1D transforms
    int idist = nCols; // number of columns in input array
    int odist = static_cast<int>(nCols); ///2+1; // number of columns in output array
    int istride = 1; int ostride = 1; // array is contiguous in memory
    int* inembed = n; int *onembed = n;
#   pragma omp critical
    {
        plan_ = fftw_plan_many_dft_c2r(
            rank, n, howmany,
            reinterpret_cast<fftw_complex*>(&in(0)), inembed, istride, idist,
            &out(0), onembed, ostride, odist,
            FFTW_ESTIMATE
        );
    }
}

template<class S>
inline
IFFT<S>::~IFFT() {
#   pragma omp critical
    fftw_destroy_plan(plan_);
}

template<class S>
inline void
IFFT<S>::execute() {
    fftw_execute(plan_);
}

} // namespace andres

#endif // #ifndef ANDRES_MARRAY_FFTW_HXX
