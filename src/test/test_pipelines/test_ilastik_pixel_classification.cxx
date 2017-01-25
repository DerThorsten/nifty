#define BOOST_TEST_MODULE pipelines_ilastik_pixel_classification

#include <tbb/tbb.h>
#include <boost/test/unit_test.hpp>
#include "nifty/pipelines/ilastik_backend/batch_prediction_task.hxx"
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
    coordinate block_shape({64,64,64});

    auto selected_features = std::make_pair(std::vector<std::string>({"GaussianSmoothing"}),
            std::vector<double>({2.,3.5}));

    batch_prediction_task<dim>& batch = *new(tbb::task::allocate_root()) batch_prediction_task<dim>(
            raw_file, "exported_data",
            rf_filename, rf_path,
            selected_features,
            block_shape, max_num_entries);
    tbb::task::spawn_root_and_wait(batch);
}

