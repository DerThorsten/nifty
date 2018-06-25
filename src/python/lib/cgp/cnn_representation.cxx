#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>



#include "nifty/python/converter.hxx"
#include "nifty/cgp/geometry.hxx"
#include "nifty/cgp/bounds.hxx"


namespace py = pybind11;



namespace nifty{
namespace cgp{



    




    void exportCnnRepresentation(py::module & module) {

        module.def("cell0Cell1Masks",
            [](
                const nifty::marray::PyView<float,   3> & padded_image_data,
                const nifty::marray::PyView<int32_t, 2> & padded_cell_1_mask,
                const nifty::marray::PyView<int32_t, 2> & padded_cell_0_coordinates,
                const nifty::marray::PyView<int32_t, 2> & cell_0_bounds,
                const uint64_t size
            ){

                NIFTY_CHECK_OP(padded_image_data.shape(1), ==, padded_cell_1_mask.shape(0),"");
                NIFTY_CHECK_OP(padded_image_data.shape(2), ==, padded_cell_1_mask.shape(1),"");

                uint64_t n_c0_3  = 0;
                uint64_t n_c0_4  = 0;
                const uint64_t n_cell_0 = cell_0_bounds.shape(0);
                for(uint64_t c0_index=0; c0_index<n_cell_0; ++c0_index)
                {
                    if(cell_0_bounds(c0_index, 3) == 0){
                        ++n_c0_3;
                    }
                    else{
                        ++n_c0_4;
                    }
                }



                const uint64_t hsize = ((size-2)/2 );



                nifty::marray::PyView<uint8_t> out3({
                    std::size_t(n_c0_3),
                    std::size_t(4),
                    std::size_t(size),
                    std::size_t(size)
                });
                nifty::marray::PyView<float> fout3({
                    std::size_t(n_c0_3),
                    std::size_t(padded_image_data.shape(0)),
                    std::size_t(size),
                    std::size_t(size)
                });
                nifty::marray::PyView<uint8_t> out4({
                    std::size_t(n_c0_4),
                    std::size_t(5),
                    std::size_t(size),
                    std::size_t(size)
                });
                nifty::marray::PyView<float> fout4({
                    std::size_t(n_c0_4),
                    std::size_t(padded_image_data.shape(0)),
                    std::size_t(size),
                    std::size_t(size)
                });
                std::fill(out3.begin(), out3.end(),0);
                std::fill(out4.begin(), out4.end(),0);
                //out = uint8_t(0);

                uint64_t c0_3_index=0;
                uint64_t c0_4_index=0;

                for(uint64_t c0_index=0; c0_index<n_cell_0; ++c0_index)
                {
                    const uint64_t jsize = cell_0_bounds(c0_index, 3) == 0 ? 3 : 4;
                    const uint64_t c0_x0 = padded_cell_0_coordinates(c0_index,0);
                    const uint64_t c0_x1 = padded_cell_0_coordinates(c0_index,1);

                    const uint64_t begin_x0 = c0_x0 - hsize;
                    const uint64_t begin_x1 = c0_x1 - hsize;
                    const uint64_t end_x0 = begin_x0 + size;
                    const uint64_t end_x1 = begin_x1 + size;

                    NIFTY_CHECK_OP(end_x0 - begin_x0, ==, size,"");
                    NIFTY_CHECK_OP(end_x1 - begin_x1, ==, size,"");

                    for(uint64_t x0=begin_x0, xx0=0; x0<end_x0; ++x0,  ++xx0)
                    for(uint64_t x1=begin_x1, xx1=0; x1<end_x1; ++x1,  ++xx1)
                    {

                        for(uint64_t c=0; c<padded_image_data.shape(0); ++c)
                        {
                            if(jsize==3)
                                fout3(c0_3_index, c, xx0, xx1) = padded_image_data(c, x0, x1);
                            else
                                fout4(c0_4_index, c, xx0, xx1) = padded_image_data(c, x0, x1);
                        }


                        const auto c1_mask_val = padded_cell_1_mask(x0, x1);
                        if(c1_mask_val != 0)
                        {
                            //bool is_own_cell1 = false;
                            for(uint64_t ji=0; ji<jsize; ++ji)
                            {
                                if(c1_mask_val == cell_0_bounds(c0_index, ji))
                                {
                                    if(jsize==3)
                                        out3(c0_3_index, ji, xx0, xx1) = 1;
                                    else
                                        out4(c0_4_index, ji, xx0, xx1) = 1;
                                    //is_own_cell1 = true;
                                    break;
                                }
                            }
                            {
                                if(jsize==3)
                                    out3(c0_3_index, jsize, xx0, xx1) = 1;
                                else
                                    out4(c0_4_index, jsize, xx0, xx1) = 1;
                            }
                        }
                    }
                    if(jsize==3)
                        ++c0_3_index;
                    else
                        ++c0_4_index;
                }
                return std::make_tuple(out3, out4, fout3, fout4);
            },

            py::arg("padded_image_data"),
            py::arg("padded_cell_1_mask"),
            py::arg("padded_cell_0_coordinates"),
            py::arg("cell_0_bounds"),
            py::arg("size")
        );
    }
}
}
