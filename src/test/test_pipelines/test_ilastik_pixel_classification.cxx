#define BOOST_TEST_MODULE pipelines_ilastik_pixel_classification


#include <boost/test/unit_test.hpp>
// #include "nifty/pipelines/ilastik_backend/batch_prediction_task.hxx"
#include "nifty/pipelines/ilastik_backend/feature_computation_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_prediction_task.hxx"
#include "nifty/pipelines/ilastik_backend/random_forest_loader.hxx"
#include "nifty/pipelines/ilastik_backend/interactive_pixel_classification.hxx"

BOOST_AUTO_TEST_CASE(PixelClassificationPredictionTest)
{   
    using namespace nifty::pipelines::ilastik_backend;
    
    constexpr size_t dim  = 3;

    using data_type = float;
    using in_data_type = uint8_t;
    using float_array = nifty::marray::Marray<data_type>;
    using float_array_view = nifty::marray::View<data_type>;
    using prediction_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    using feature_cache = tbb::concurrent_lru_cache<size_t, float_array_view, std::function<float_array_view(size_t)>>;
    constexpr size_t max_num_entries = 100;
    using raw_cache = Hdf5Input<in_data_type, dim, false, in_data_type>;
    using coordinate = nifty::array::StaticArray<int64_t, dim>;


    // load random forests
    const std::string rf_filename = "./testPC.ilp";
    const std::string rf_path = "/PixelClassification/ClassifierForests/Forest";
    RandomForestVectorType rf_vector;
    get_rfs_from_file(rf_vector, rf_filename, rf_path, 4);

    std::string raw_file = "./testraw.h5";
    auto file = nifty::hdf5::openFile(raw_file);
    nifty::hdf5::Hdf5Array<in_data_type> raw_array(file, "exported_data");
    raw_cache rc(raw_array);
    coordinate fake_shape({128,128,128});
    coordinate begin({0,0,0});
    coordinate blockShape({64,64,64});
    nifty::tools::Blocking<dim> blocking(begin, fake_shape, blockShape);

    auto selected_features = std::make_pair(std::vector<std::string>({"GaussianSmoothing"}),
            std::vector<double>({2.,3.}));

    // batch_prediction_task batch_pred();
    // tbb::spawn_root_and_wait(batch_pred);
}

