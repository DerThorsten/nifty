#ifndef _RANDOM_FOREST_LOADER_HXX_
#define _RANDOM_FOREST_LOADER_HXX_

#include <vigra/random_forest_hdf5_impex.hxx>
#include <hdf5_hl.h>

namespace nifty
{
    namespace pipelines
    {
        namespace ilastik_backend
        {
            typedef size_t LabelType;
            typedef vigra::RandomForest<LabelType> RandomForestType;
            typedef std::vector<RandomForestType> RandomForestVectorType;

            /** 
             * @brief Read the random forests from the hdf5 files.
             * 
             * WARNING: this shows some warnings on the command line because we try to read one more
             *          tree than is available. But that seems to be the easiest option to get all RFs in the group.
             * 
             */
            bool get_rfs_from_file(
                RandomForestVectorType& random_forests_vector,
                const std::string& filename,
                const std::string& path_in_file = "PixelClassification/ClassifierForests/Forest",
                int n_leading_zeros = 4);
        }
    }
}

#endif // _RANDOM_FOREST_LOADER_HXX_
