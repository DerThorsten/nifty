#include <iostream>
#include <tuple>
#include <random>

#include "nifty/tools/runtime_check.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "nifty/parallel/threadpool.hxx"

// test data generated with ff pybindings

//
// 2d filters
//
typedef typename xt::xarray<float>::shape_type ShapeType;

//std::tuple<std::vector<std::vector<float>>> get2DTestData() {
auto get2DTestData() {

    // test data generated with ff pybindings

    //
    // 2d filters
    //

    std::vector<std::vector<float>> gaussian2D(5, std::vector<float>());
    gaussian2D[0] = std::vector<float>( {0.01166641,  0.02662143,  0.04310189,  0.02662143,  0.01166641} );
    gaussian2D[1] = std::vector<float>( {0.02662143,  0.0607471 ,  0.09835363,  0.0607471 ,  0.02662143} );
    gaussian2D[2] = std::vector<float>( {0.04310189,  0.09835364,  0.15924114,  0.09835364,  0.04310189} );
    gaussian2D[3] = std::vector<float>( {0.02662143,  0.0607471 ,  0.09835363,  0.0607471 ,  0.02662143} );
    gaussian2D[4] = std::vector<float>( {0.01166641,  0.02662143,  0.04310189,  0.02662143,  0.01166641} );

    std::vector<std::vector<float>> laplacian2D(5, std::vector<float>());
    laplacian2D[0] = std::vector<float>( {0.07002892,  0.0837328 ,  0.08667994,  0.0837328 ,  0.07002892} );
    laplacian2D[1] = std::vector<float>( {0.0837328 ,  0.0174964 , -0.08323152,  0.0174964 ,  0.0837328 } );
    laplacian2D[2] = std::vector<float>( {0.08667994, -0.08323152, -0.3153795 , -0.08323152,  0.08667994} );
    laplacian2D[3] = std::vector<float>( {0.0837328 ,  0.0174964 , -0.08323152,  0.0174964 ,  0.0837328 } );
    laplacian2D[4] = std::vector<float>( {0.07002892,  0.0837328 ,  0.08667994,  0.0837328 ,  0.07002892} );

    std::vector<std::vector<float>> hessian02D(5, std::vector<float>());
    hessian02D[0] = std::vector<float>( {0.03501446,  0.07989904,  0.12936191,  0.07989904,  0.03501446} );
    hessian02D[1] = std::vector<float>( {0.07989904,  0.06104838,  0.01416392,  0.06104838,  0.07989904} );
    hessian02D[2] = std::vector<float>( {0.12936191,  0.01416392, -0.15768975,  0.01416392,  0.12936191} );
    hessian02D[3] = std::vector<float>( {0.07989904,  0.06104838,  0.01416392,  0.06104838,  0.07989904} );
    hessian02D[4] = std::vector<float>( {0.03501446,  0.07989904,  0.12936191,  0.07989904,  0.03501446} );

    std::vector<std::vector<float>> hessian12D(5, std::vector<float>());
    hessian12D[0] = std::vector<float>( { 0.03501446,  0.00383376, -0.04268197,  0.00383376,  0.03501446} );
    hessian12D[1] = std::vector<float>( { 0.00383376, -0.04355198, -0.09739543, -0.04355198,  0.00383376} );
    hessian12D[2] = std::vector<float>( {-0.04268197, -0.09739544, -0.15768975, -0.09739544, -0.04268197} );
    hessian12D[3] = std::vector<float>( { 0.00383376, -0.04355198, -0.09739543, -0.04355198,  0.00383376} );
    hessian12D[4] = std::vector<float>( { 0.03501446,  0.00383376, -0.04268197,  0.00383376,  0.03501446} );

    return std::make_tuple(gaussian2D, laplacian2D, hessian02D, hessian12D);
}


// 3d test data
//std::tuple<std::vector<std::vector<std::vector<float>>>> get3DTestData() {
auto get3DTestData() {

    std::vector<std::vector<std::vector<float>>> gaussian3D(5, std::vector<std::vector<float>>(5, std::vector<float>()));
    gaussian3D[0][0] = std::vector<float>( {0.0012601 ,  0.00287541,  0.00465549,  0.00287541,  0.0012601 } );
    gaussian3D[0][1] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );
    gaussian3D[0][2] = std::vector<float>( {0.00465549,  0.01062329,  0.01719982,  0.01062329,  0.00465549} );
    gaussian3D[0][3] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );
    gaussian3D[0][4] = std::vector<float>( {0.0012601 ,  0.00287541,  0.00465549,  0.00287541,  0.0012601 } );

    gaussian3D[1][0] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );
    gaussian3D[1][1] = std::vector<float>( {0.00656136,  0.01497229,  0.02424115,  0.01497229,  0.00656136} );
    gaussian3D[1][2] = std::vector<float>( {0.01062329,  0.02424115,  0.03924805,  0.02424115,  0.01062329} );
    gaussian3D[1][3] = std::vector<float>( {0.00656136,  0.01497229,  0.02424115,  0.01497229,  0.00656136} );
    gaussian3D[1][4] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );

    gaussian3D[2][0] = std::vector<float>( {0.00465549,  0.01062329,  0.01719982,  0.01062329,  0.00465549} );
    gaussian3D[2][1] = std::vector<float>( {0.01062329,  0.02424115,  0.03924805,  0.02424115,  0.01062329} );
    gaussian3D[2][2] = std::vector<float>( {0.01719982,  0.03924805,  0.06354523,  0.03924805,  0.01719982} );
    gaussian3D[2][3] = std::vector<float>( {0.01062329,  0.02424115,  0.03924805,  0.02424115,  0.01062329} );
    gaussian3D[2][4] = std::vector<float>( {0.00465549,  0.01062329,  0.01719982,  0.01062329,  0.00465549} );

    gaussian3D[3][0] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );
    gaussian3D[3][1] = std::vector<float>( {0.00656136,  0.01497229,  0.02424115,  0.01497229,  0.00656136} );
    gaussian3D[3][2] = std::vector<float>( {0.01062329,  0.02424115,  0.03924805,  0.02424115,  0.01062329} );
    gaussian3D[3][3] = std::vector<float>( {0.00656136,  0.01497229,  0.02424115,  0.01497229,  0.00656136} );
    gaussian3D[3][4] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );

    gaussian3D[4][0] = std::vector<float>( {0.0012601 ,  0.00287541,  0.00465549,  0.00287541,  0.0012601 } );
    gaussian3D[4][1] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );
    gaussian3D[4][2] = std::vector<float>( {0.00465549,  0.01062329,  0.01719982,  0.01062329,  0.00465549} );
    gaussian3D[4][3] = std::vector<float>( {0.00287541,  0.00656136,  0.01062329,  0.00656136,  0.00287541} );
    gaussian3D[4][4] = std::vector<float>( {0.0012601 ,  0.00287541,  0.00465549,  0.00287541,  0.0012601 } );

    std::vector<std::vector<std::vector<float>>> laplacian3D(5, std::vector<std::vector<float>>(5, std::vector<float>()));
    laplacian3D[0][0] = std::vector<float>( {0.01134586,  0.01767407,  0.02333493,  0.01767407,  0.01134586} );
    laplacian3D[0][1] = std::vector<float>( {0.01767407,  0.02158247,  0.0228938 ,  0.02158247,  0.01767407} );
    laplacian3D[0][2] = std::vector<float>( {0.02333493,  0.0228938 ,  0.0175574 ,  0.0228938 ,  0.02333493} );
    laplacian3D[0][3] = std::vector<float>( {0.01767407,  0.02158247,  0.0228938 ,  0.02158247,  0.01767407} );
    laplacian3D[0][4] = std::vector<float>( {0.01134586,  0.01767407,  0.02333493,  0.01767407,  0.01134586} );

    laplacian3D[1][0] = std::vector<float>( {0.01767407,  0.02158247,  0.02289381,  0.02158247,  0.01767407} );
    laplacian3D[1][1] = std::vector<float>( {0.02158247,  0.00646849, -0.01702304,  0.00646849,  0.02158247} );
    laplacian3D[1][2] = std::vector<float>( {0.0228938 , -0.01702304, -0.07207924, -0.01702304,  0.0228938 } );
    laplacian3D[1][3] = std::vector<float>( {0.02158247,  0.00646849, -0.01702304,  0.00646849,  0.02158247} );
    laplacian3D[1][4] = std::vector<float>( {0.01767407,  0.02158247,  0.02289381,  0.02158247,  0.01767407} );

    laplacian3D[2][0] = std::vector<float>( {0.02333493,  0.0228938 ,  0.0175574 ,  0.0228938 ,  0.02333493} );
    laplacian3D[2][1] = std::vector<float>( {0.0228938 , -0.01702304, -0.07207924, -0.01702304,  0.0228938 } );
    laplacian3D[2][2] = std::vector<float>( {0.0175574 , -0.07207924, -0.18877843, -0.07207924,  0.0175574 } );
    laplacian3D[2][3] = std::vector<float>( {0.0228938 , -0.01702304, -0.07207924, -0.01702304,  0.0228938 } );
    laplacian3D[2][4] = std::vector<float>( {0.02333493,  0.0228938 ,  0.0175574 ,  0.0228938 ,  0.02333493} );

    laplacian3D[3][0] = std::vector<float>( {0.01767407,  0.02158247,  0.02289381,  0.02158247,  0.01767407} );
    laplacian3D[3][1] = std::vector<float>( {0.02158247,  0.00646849, -0.01702304,  0.00646849,  0.02158247} );
    laplacian3D[3][2] = std::vector<float>( {0.0228938 , -0.01702304, -0.07207924, -0.01702304,  0.0228938 } );
    laplacian3D[3][3] = std::vector<float>( {0.02158247,  0.00646849, -0.01702304,  0.00646849,  0.02158247} );
    laplacian3D[3][4] = std::vector<float>( {0.01767407,  0.02158247,  0.02289381,  0.02158247,  0.01767407} );

    laplacian3D[4][0] = std::vector<float>( {0.01134586,  0.01767407,  0.02333493,  0.01767407,  0.01134586} );
    laplacian3D[4][1] = std::vector<float>( {0.01767407,  0.02158247,  0.0228938 ,  0.02158247,  0.01767407} );
    laplacian3D[4][2] = std::vector<float>( {0.02333493,  0.0228938 ,  0.0175574 ,  0.0228938 ,  0.02333493} );
    laplacian3D[4][3] = std::vector<float>( {0.01767407,  0.02158247,  0.0228938 ,  0.02158247,  0.01767407} );
    laplacian3D[4][4] = std::vector<float>( {0.01134586,  0.01767407,  0.02333493,  0.01767407,  0.01134586} );

    std::vector<std::vector<std::vector<float>>> hessian3D0(5, std::vector<std::vector<float>>(5, std::vector<float>()));
    hessian3D0[0][0] = std::vector<float>( {0.00378195,  0.00863124,  0.01397253,  0.00863124,  0.00378195} );
    hessian3D0[0][1] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863124} );
    hessian3D0[0][2] = std::vector<float>( {0.01397253,  0.03188374,  0.05162191,  0.03188374,  0.01397253} );
    hessian3D0[0][3] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863124} );
    hessian3D0[0][4] = std::vector<float>( {0.00378195,  0.00863124,  0.01397253,  0.00863124,  0.00378195} );

    hessian3D0[1][0] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863124} );
    hessian3D0[1][1] = std::vector<float>( {0.01969266,  0.02793694,  0.02436137,  0.02793694,  0.01969266} );
    hessian3D0[1][2] = std::vector<float>( {0.03188374,  0.02436137,  0.00565212,  0.02436137,  0.03188374} );
    hessian3D0[1][3] = std::vector<float>( {0.01969266,  0.02793694,  0.02436137,  0.02793694,  0.01969266} );
    hessian3D0[1][4] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863172} );

    hessian3D0[2][0] = std::vector<float>( {0.01397468,  0.03188374,  0.05162191,  0.03188374,  0.01397468} );
    hessian3D0[2][1] = std::vector<float>( {0.03188374,  0.02436137,  0.00565212,  0.02436137,  0.03188374} );
    hessian3D0[2][2] = std::vector<float>( {0.05162191,  0.00565212, -0.06292614,  0.00565212,  0.05162191} );
    hessian3D0[2][3] = std::vector<float>( {0.03188374,  0.02436137,  0.00565212,  0.02436137,  0.03188374} );
    hessian3D0[2][4] = std::vector<float>( {0.01397468,  0.03188374,  0.05162191,  0.03188374,  0.01397501} );

    hessian3D0[3][0] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863124} );
    hessian3D0[3][1] = std::vector<float>( {0.01969266,  0.02793694,  0.02436137,  0.02793694,  0.01969266} );
    hessian3D0[3][2] = std::vector<float>( {0.03188374,  0.02436137,  0.00565212,  0.02436137,  0.03188374} );
    hessian3D0[3][3] = std::vector<float>( {0.01969266,  0.02793694,  0.02436137,  0.02793694,  0.01969266} );
    hessian3D0[3][4] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863172} );

    hessian3D0[4][0] = std::vector<float>( {0.00378195,  0.00863124,  0.01397253,  0.00863124,  0.00378195} );
    hessian3D0[4][1] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863124} );
    hessian3D0[4][2] = std::vector<float>( {0.01397253,  0.03188374,  0.05162191,  0.03188374,  0.01397253} );
    hessian3D0[4][3] = std::vector<float>( {0.00863124,  0.01969266,  0.03188374,  0.01969266,  0.00863124} );
    hessian3D0[4][4] = std::vector<float>( {0.00378195,  0.00863124,  0.01397253,  0.00863124,  0.00378195} );

    return std::make_tuple(gaussian3D, laplacian3D, hessian3D0);

}

BOOST_AUTO_TEST_CASE(FastfiltersWrapperTest2D)
{

    ShapeType shapeIn = {5,5};
    xt::xarray<float> in(shapeIn);
    std::fill(in.begin(), in.end(), 0.);
    in(2,2) = 1.;

    using namespace nifty::features;
    typedef typename ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    // fastfilters segfault for larger sigmas for a 5x5 array
    std::vector<double> sigmas({1.});
    FiltersToSigmasType filtersToSigmas({ { true },      // GaussianSmoothing
                                          { true },      // LaplacianOfGaussian
                                          { false},   // GaussianGradientMagnitude
                                          { true } });  // HessianOfGaussianEigenvalues

    ApplyFilters<2> functor(sigmas, filtersToSigmas);

    ShapeType shapeOut = {functor.numberOfChannels(), shapeIn[0], shapeIn[1]};
    xt::xarray<float> out(shapeOut);

    functor(in, out);

    // test shapes
    NIFTY_TEST_OP(out.shape(1),==,shapeIn[0])
    NIFTY_TEST_OP(out.shape(2),==,shapeIn[1])
    NIFTY_TEST_OP(out.shape(0),==,functor.numberOfChannels())

    auto testData = get2DTestData();

    // test filter responses for correctnes for first sigma val
    for(size_t y = 0; y < in.shape(0); y++) {
        for(size_t x = 0; x < in.shape(1); x++) {
            NIFTY_CHECK_EQ_TOL(out(0,y,x),std::get<0>(testData)[y][x],1e-6)
            NIFTY_CHECK_EQ_TOL(out(1,y,x),std::get<1>(testData)[y][x],1e-6)
            NIFTY_CHECK_EQ_TOL(out(2,y,x),std::get<2>(testData)[y][x],1e-6)
            NIFTY_CHECK_EQ_TOL(out(3,y,x),std::get<3>(testData)[y][x],1e-6)
        }
    }

}


BOOST_AUTO_TEST_CASE(FastfiltersWrapperTest2DPresmooth)
{

    // create random test data
    ShapeType shapeIn = {100,100};
    xt::xarray<float> in(shapeIn);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr(0.,1.);
    auto draw = std::bind( distr, generator );

    for(size_t ii = 0; ii < in.shape(0); ++ii) {
        for(size_t jj = 0; jj < in.shape(1); ++jj)
            in(ii,jj) = draw();
    }

    using namespace nifty::features;
    typedef typename ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    // fastfilters segfault for larger sigmas for a 5x5 array
    std::vector<double> sigmas({1.,2.,5.,8.});
    FiltersToSigmasType filtersToSigmas({ { true, true, true, true },      // GaussianSmoothing
                                          { true, true, true, true },      // LaplacianOfGaussian
                                          { true, true, true, true },   // GaussianGradientMagnitude
                                          { true, true, true, true } });  // HessianOfGaussianEigenvalues

    ApplyFilters<2> functor(sigmas, filtersToSigmas);

    ShapeType shapeOut = {functor.numberOfChannels(), shapeIn[0], shapeIn[1]};
    xt::xarray<float> outExpected(shapeOut);
    xt::xarray<float> outActual(shapeOut);

    functor(in, outExpected, true);
    functor(in, outActual, false);

    // test shapes
    for(int d = 0; d < outExpected.dimension(); ++d)
        NIFTY_TEST_OP(outActual.shape(d),==,outExpected.shape(d))
    NIFTY_TEST_OP(outActual.shape(0),==,functor.numberOfChannels())

    // test filter responses for correctnes for first sigma val
    double tolerance = 1e-1; // we pass only if we allow for first tolerance after the first comma
    for(size_t c = 0; c < outActual.shape(0); ++c) {
        for(size_t x = 0; x < outActual.shape(1); ++x) {
            for(size_t y = 0; y < outActual.shape(2); ++y) {
                NIFTY_CHECK_EQ_TOL(outActual(c,x,y),outExpected(c,x,y),tolerance);
            }
        }
    }
}


BOOST_AUTO_TEST_CASE(FastfiltersWrapperTest2DParallel)
{

    ShapeType shapeIn = {5,5};
    xt::xarray<float> in(shapeIn);
    std::fill(in.begin(), in.end(), 0.);
    in(2,2) = 1.;

    using namespace nifty::features;
    typedef typename ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    // fastfilters segfault for larger sigmas for a 5x5 array
    std::vector<double> sigmas({1.});
    FiltersToSigmasType filtersToSigmas({ { true },      // GaussianSmoothing
                                          { true },      // LaplacianOfGaussian
                                          { false},   // GaussianGradientMagnitude
                                          { true } });  // HessianOfGaussianEigenvalues

    ApplyFilters<2> functor(sigmas, filtersToSigmas);

    ShapeType shapeOut = {functor.numberOfChannels(),shapeIn[0],shapeIn[1]};
    xt::xarray<float> out(shapeOut);

    nifty::parallel::ParallelOptions pOpts(-1);
    nifty::parallel::ThreadPool threadpool(pOpts);

    functor(in, out, threadpool);

    // test shapes
    NIFTY_TEST_OP(out.shape(1),==,shapeIn[0])
    NIFTY_TEST_OP(out.shape(2),==,shapeIn[1])
    NIFTY_TEST_OP(out.shape(0),==,functor.numberOfChannels())

    auto testData = get2DTestData();

    // test filter responses for correctnes for first sigma val
    for(size_t y = 0; y < in.shape(0); y++) {
        for(size_t x = 0; x < in.shape(1); x++) {
            NIFTY_CHECK_EQ_TOL(out(0,y,x),std::get<0>(testData)[y][x],1e-6)
            NIFTY_CHECK_EQ_TOL(out(1,y,x),std::get<1>(testData)[y][x],1e-6)
            NIFTY_CHECK_EQ_TOL(out(2,y,x),std::get<2>(testData)[y][x],1e-6)
            NIFTY_CHECK_EQ_TOL(out(3,y,x),std::get<3>(testData)[y][x],1e-6)
        }
    }

}


BOOST_AUTO_TEST_CASE(FastfiltersWrapperTest3D)
{
    ShapeType shapeIn = {5,5,5};
    xt::xarray<float> in(shapeIn);
    std::fill(in.begin(), in.end(), 0.);
    in(2,2,2) = 1.;

    using namespace nifty::features;
    typedef typename ApplyFilters<3>::FiltersToSigmasType FiltersToSigmasType;

    // fastfilters segfault for larger sigmas for a 5x5 array
    std::vector<double> sigmas({1.});
    FiltersToSigmasType filtersToSigmas({ { true },      // GaussianSmoothing
                                          { true },      // LaplacianOfGaussian
                                          { false},   // GaussianGradientMagnitude
                                          { true } });  // HessianOfGaussianEigenvalues

    ApplyFilters<3> functor(sigmas, filtersToSigmas);

    ShapeType shapeOut = {functor.numberOfChannels(),shapeIn[0],shapeIn[1],shapeIn[2]};
    xt::xarray<float> out(shapeOut);

    functor(in, out);

    // test shapes
    NIFTY_TEST_OP(out.shape(1),==,shapeIn[0])
    NIFTY_TEST_OP(out.shape(2),==,shapeIn[1])
    NIFTY_TEST_OP(out.shape(3),==,shapeIn[2])
    NIFTY_TEST_OP(out.shape(0),==,functor.numberOfChannels())

    auto testData = get3DTestData();

    // test filter responses for correctnes for first sigma val
    for(size_t z = 0; z < in.shape(0); z++) {
        for(size_t y = 0; y < in.shape(1); y++) {
            for(size_t x = 0; x < in.shape(2); x++) {
                NIFTY_CHECK_EQ_TOL(out(0,z,y,x),std::get<0>(testData)[z][y][x],1e-6)
                NIFTY_CHECK_EQ_TOL(out(1,z,y,x),std::get<1>(testData)[z][y][x],1e-6)
                NIFTY_CHECK_EQ_TOL(out(2,z,y,x),std::get<2>(testData)[z][y][x],1e-5)
            }
        }
    }

}


BOOST_AUTO_TEST_CASE(FastfiltersWrapperTest3DPresmooth)
{

    // create random test data
    ShapeType shapeIn = {100,100,100};
    xt::xarray<float> in(shapeIn);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr(0.,1.);
    auto draw = std::bind( distr, generator );

    for(size_t ii = 0; ii < in.shape(0); ++ii) {
        for(size_t jj = 0; jj < in.shape(1); ++jj){
            for(size_t kk = 0; kk < in.shape(2); ++kk)
                in(ii,jj,kk) = draw();
        }
    }

    using namespace nifty::features;
    typedef typename ApplyFilters<3>::FiltersToSigmasType FiltersToSigmasType;

    std::vector<double> sigmas({1.,2.,5.,8.});
    FiltersToSigmasType filtersToSigmas({ { true, true, true, true },      // GaussianSmoothing
                                          { true, true, true, true },      // LaplacianOfGaussian
                                          { true, true, true, true },      // GaussianGradientMagnitude
                                          { true, true, true, true } });   // HessianOfGaussianEigenvalues

    ApplyFilters<3> functor(sigmas, filtersToSigmas);

    ShapeType shapeOut = {functor.numberOfChannels(),shapeIn[0],shapeIn[1],shapeIn[2]};
    xt::xarray<float> outExpected(shapeOut);
    xt::xarray<float> outActual(shapeOut);

    functor(in, outExpected, true);
    functor(in, outActual, false);

    // test shapes
    for(int d = 0; d < outExpected.dimension(); ++d)
        NIFTY_TEST_OP(outActual.shape(d),==,outExpected.shape(d))
    NIFTY_TEST_OP(outActual.shape(0),==,functor.numberOfChannels())

    // test filter responses for correctnes for first sigma val
    double tolerance = 1e-2; // we pass only if we allow for first tolerance after the first comma
    // apparently this is more robust in 3d than in 2d
    for(size_t c = 0; c < outActual.shape(0); ++c) {
        for(size_t x = 0; x < outActual.shape(1); ++x) {
            for(size_t y = 0; y < outActual.shape(2); ++y) {
                for(size_t z = 0; z < outActual.shape(3); ++z) {
                    NIFTY_CHECK_EQ_TOL(outActual(c,x,y,z),outExpected(c,x,y,z),tolerance);
                }
            }
        }
    }
}


BOOST_AUTO_TEST_CASE(FastfiltersWrapperTest3DParallel)
{

    ShapeType shapeIn = {5,5,5};
    xt::xarray<float> in(shapeIn);
    std::fill(in.begin(), in.end(), 0.);
    in(2,2,2) = 1.;

    using namespace nifty::features;
    typedef typename ApplyFilters<3>::FiltersToSigmasType FiltersToSigmasType;

    // fastfilters segfault for larger sigmas for a 5x5 array
    std::vector<double> sigmas({1.});
    FiltersToSigmasType filtersToSigmas { { true },      // GaussianSmoothing
                                          { true },      // LaplacianOfGaussian
                                          { false},    // GaussianGradientMagnitude
                                          { true } };  // HessianOfGaussianEigenvalues

    ApplyFilters<3> functor(sigmas, filtersToSigmas);

    ShapeType shapeOut = {functor.numberOfChannels(),shapeIn[0],shapeIn[1],shapeIn[2]};
    xt::xarray<float> out(shapeOut.begin(), shapeOut.end());

    nifty::parallel::ParallelOptions pOpts(-1);
    nifty::parallel::ThreadPool threadpool(pOpts);

    functor(in, out, threadpool);

    // test shapes
    NIFTY_TEST_OP(out.shape(1),==,shapeIn[0])
    NIFTY_TEST_OP(out.shape(2),==,shapeIn[1])
    NIFTY_TEST_OP(out.shape(3),==,shapeIn[2])
    NIFTY_TEST_OP(out.shape(0),==,functor.numberOfChannels())

    auto testData = get3DTestData();

    // test filter responses for correctnes for first sigma val
    for(size_t z = 0; z < in.shape(0); z++) {
        for(size_t y = 0; y < in.shape(1); y++) {
            for(size_t x = 0; x < in.shape(2); x++) {
                NIFTY_CHECK_EQ_TOL(out(0,z,y,x),std::get<0>(testData)[z][y][x],1e-6)
                NIFTY_CHECK_EQ_TOL(out(1,z,y,x),std::get<1>(testData)[z][y][x],1e-6)
                NIFTY_CHECK_EQ_TOL(out(2,z,y,x),std::get<2>(testData)[z][y][x],1e-5)
            }
        }
    }
}

int main(){
    fastfiltersWrapperTest2D();
    fastfiltersWrapperTest3D();
}
