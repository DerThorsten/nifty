


namespace nifty{
namespace pipelines{
namespace ilastik_backend{

    template<size_t DIM>
    class SpatialTag{
    public:
        typedef std::integral_constant<size_t , DIM> DimensionType;
        typedef std::integral_constant<size_t , DIM> SpaceTimeDimensions;
    };

    template<size_t DIM>
    class SpatialWithTimeTag{
    public:
        typedef std::integral_constant<size_t , DIM> DimensionType;
        typedef std::integral_constant<size_t , DIM> SpaceTimeDimensions;
    };  


    //template< size_t N_SPACIAL_DIM, bool HAS_TIME>


}
}
}