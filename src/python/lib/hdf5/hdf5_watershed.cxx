#ifdef WITH_HDF5
#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/hdf5/hdf5_array.hxx"

#include "vigra/multi_array_chunked_hdf5.hxx"
#include "vigra/blockwise_watersheds.hxx"


namespace py = pybind11;



namespace nifty{
namespace hdf5{

    template<class T, class L, unsigned int DIM>
    void exportHdf5BlockwiseWatershedT(py::module & hdf5Module, const std::string & fnamePostfix) {

        const std::string fname = std::string("blockwiseWatershed_") + fnamePostfix;
        hdf5Module.def(fname.c_str(),
            [](
                std::string fData,
                std::string dData,
                std::string fLabels,
                std::string dLabels,
                std::array< unsigned int, DIM> blockShapeWatersheds,
                std::array< unsigned int, DIM> blockShapeArray,
                int numberOfThreads
            ){
                typedef vigra::ChunkedArrayHDF5<DIM, T> DataArray;
                typedef vigra::ChunkedArrayHDF5<DIM, L> LabelsArray;
                typedef typename LabelsArray::shape_type VigraShapeType;



                //VigraShapeType chunkShape;
                //  std::fill(vigraChunkedArrayChunkShape.begin())
        

                std::cout<<"data array\n";
                vigra::HDF5File h5FileData(fData,  vigra::HDF5File::OpenReadOnly );
                DataArray dataArray(h5FileData,dData,
                    vigra::HDF5File::OpenReadOnly);

                std::cout<<"labels array\n";

                vigra::ChunkedArrayOptions cOpts;
                cOpts.compression(vigra::ZLIB_FAST);


                vigra::HDF5File h5FileLabels(fLabels,  vigra::HDF5File::Open );
                LabelsArray labelsArray(h5FileLabels,dLabels,
                    vigra::HDF5File::Open, dataArray.shape(),
                    VigraShapeType(128), cOpts);

                vigra::unionFindWatershedsBlockwise(dataArray, labelsArray);
               
            }
        )
        ;
    }


    void exportHdf5BlockwiseWatershed(py::module & hdf5Module) {
        exportHdf5BlockwiseWatershedT<float, uint32_t, 3>(hdf5Module, "float32_uint32_3d");
    }

}
}

#endif
