#include "nifty/tools/blocking.hxx"



void blockingTest()
{   

    typedef nifty::tools::Blocking<3> Blocking;
    typedef typename Blocking::VectorType VectorType;

    VectorType roiBegin{0,0,0}, roiEnd{100,100,100}, blockShape{10,10,10}, blockShift{0,0,0}, halo{10,10,10};

    Blocking blocking(roiBegin, roiEnd, blockShape, blockShift);


    auto blockWithHalo = blocking.getBlockWithHalo(0, halo);

    const auto & innerBlock = blockWithHalo.innerBlock();
    const auto & outerBlock = blockWithHalo.outerBlock();

    NIFTY_TEST_OP(outerBlock.end()[0],==,20);
    NIFTY_TEST_OP(innerBlock.end()[0],==,10);

}

int main() {
	blockingTest();
}