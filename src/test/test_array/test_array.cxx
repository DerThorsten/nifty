#include <iostream> 
#include <typeinfo>
#include <typeindex>
#include <sstream>
#include <vector>
#include <array>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/runtime_check.hxx"


void testArray()
{
    
    typedef nifty::array::ArrayExtender<std::vector<int> > IntVec;
    IntVec aVec = {4,1,2,3};
    IntVec bVec = {2,3,2,5};

    aVec += bVec;

    NIFTY_TEST_OP(aVec[0],==,6);
    NIFTY_TEST_OP(aVec[1],==,4);
    NIFTY_TEST_OP(aVec[2],==,4);
    NIFTY_TEST_OP(aVec[3],==,8);

    aVec/=bVec;
    aVec-=bVec;
    aVec*=bVec;
    aVec+=2;
    aVec+=4;

    std::cout<<aVec;
}

void testStaticArray()
{
    
    typedef nifty::array::ArrayExtender<nifty::array::StaticArray<int,2> > IntVec;
    IntVec aVec({int(4),int(1)});
    IntVec bVec({int(2),int(3)});

    aVec += bVec;

    NIFTY_TEST_OP(aVec[0],==,6);
    NIFTY_TEST_OP(aVec[1],==,4);


    aVec/=bVec;
    aVec-=bVec;
    aVec*=bVec;
    std::cout<<bVec;
}

void testArrayCoutOperator()
{
    
    typedef nifty::array::ArrayExtender<nifty::array::StaticArray<int,2> > IntVec;
    IntVec aVec({int(4),int(1)});
    IntVec bVec({int(2),int(3)});

    std::stringstream ss;
    ss<<aVec;
    std::cout<<ss.str();
}

void testStaticArrayConstructor()
{
    typedef nifty::array::StaticArray<int64_t, 3> Coord;
    Coord c(-2);

    NIFTY_TEST_OP(c[0],==,-2);
    NIFTY_TEST_OP(c[1],==,-2);
    NIFTY_TEST_OP(c[2],==,-2);
}

int main(){

    testArray();
    testStaticArray();
    testArrayCoutOperator();
    testStaticArrayConstructor();
}