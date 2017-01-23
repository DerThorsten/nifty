


namespace nifty{
namespace pipelines{
namespace ilastik_backend{

    template<size_t DIM>
    class SpatialTag{
    public:
        typedef std::integral_constant<size_t , DIM> DimensionType;
    };

    template<size_t DIM>
    class SpatialWithTimeTag{
    public:
        typedef std::integral_constant<size_t , DIM> DimensionType;
    };  


}
}
}