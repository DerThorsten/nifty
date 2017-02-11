#include <iostream>
#include "nifty/tools/timer.hxx"
#include "nifty/marray/marray.hxx"
#include "vigra/multi_array.hxx"



int main( int argc , char *argv[] ){

    
    
    std::vector<size_t> shape({100000000});
    nifty::marray::Marray<float> a(shape.begin(), shape.end(),0, nifty::marray::LastMajorOrder);
    nifty::marray::View<float>   v = a;

    nifty::tools::VerboseTimer timer(true);


    {
        const auto p = &v(0);
        timer.startAndPrint("ptr");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += p[i];
        }
        timer.stopAndPrint().reset();
        std::cout<<"s "<<s<<"\n";
    }

    {
    
        timer.startAndPrint("marray");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += a(i);
        }
        timer.stopAndPrint().reset();
        std::cout<<"s "<<s<<"\n";
    }

    
    {
    
        timer.startAndPrint("marrayview");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += v(i);
        }
        timer.stopAndPrint().reset();
        std::cout<<"s "<<s<<"\n";
    }

   
    {
        auto p = &v(0);
        const auto vShape = vigra::Shape1(shape[0]);
        vigra::MultiArrayView<1,float> vv(vShape, p);
        
        timer.startAndPrint("vigra");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += vv(i);
        }
        timer.stopAndPrint().reset();
        std::cout<<"s "<<s<<"\n";
    }








    std::cout<<"write\n\n";


    {
        const auto p = &v(0);
        timer.startAndPrint("ptr");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            p[i] = s++;
        }
        timer.stopAndPrint().reset();
        std::cout<<p[0]<<"\n";
    }

    {
    
        timer.startAndPrint("marray");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            a(i) = s++;
        }
        timer.stopAndPrint().reset();
        std::cout<<a(0)<<"\n";
    }

    
    {
    
        timer.startAndPrint("marrayview");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            v(i) = s++;
        }
        timer.stopAndPrint().reset();
        std::cout<<v(0)<<"\n";
    }

   
    {
        auto p = &v(0);
        const auto vShape = vigra::Shape1(shape[0]);
        vigra::MultiArrayView<1,float> vv(vShape, p);
        
        timer.startAndPrint("vigra");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            vv(i) = s++;
        }
        timer.stopAndPrint().reset();
        std::cout<<vv(0)<<"\n";
    }



}