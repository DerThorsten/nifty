#include <iostream>
#include "nifty/tools/timer.hxx"
#include "nifty/marray/marray.hxx"
#include "vigra/multi_array.hxx"

inline void marray1D(nifty::marray::View<int> & view) {
    for( int i = 0; i < view.shape(0); ++i ) {
        view(i) = i;
    }
}

inline void vigra1D(vigra::MultiArray<1,int> & varray) {
    for( int i = 0; i < varray.shape(0); ++i ) {
        varray(i) = i;
    }
}

void bench1dWrite(const int N) {
    
    size_t shape[] = {long(1e8)};
    nifty::marray::Marray<int> array(shape, shape+1);

    std::cout << "Timeing 1d accces..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int _ = 0; _ < N; ++_)
        marray1D(array);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto dur0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "... in total: " << dur0.count() << " ms" << std::endl;
    std::cout << "... per itearion " << dur0.count() / N << " ms" << std::endl;
    
    vigra::Shape1 vshape(1e8);
    vigra::MultiArray<1,int> varray(vshape);
    
    std::cout << "Timeing 1D acces Vigra..." << std::endl;
    auto t00 = std::chrono::high_resolution_clock::now();
    for(int _ = 0; _ < N; ++_)
        vigra1D(varray);
    auto t10 = std::chrono::high_resolution_clock::now();

    auto dur00 = std::chrono::duration_cast<std::chrono::milliseconds>(t10 - t00);
    std::cout << "... in total: " << dur00.count() << " ms" << std::endl;
    std::cout << "... per itearion " << dur00.count() / N << " ms" << std::endl;
}


void benchThorsten() {
    
    std::vector<size_t> shape({100000000});
    std::vector<size_t> start({0});

    nifty::marray::Marray<float> a(shape.begin(), shape.end(), 0);//, nifty::marray::LastMajorOrder);
    nifty::marray::View<float>   v = a.view(start.begin(), shape.begin());

    nifty::tools::VerboseTimer timer(true);

    {
        const auto p = &v(0);
        timer.startAndPrint("ptr");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += p[i];
        }
        timer.stopAndPrint();
        timer.reset();
        std::cout<<"s "<<s<<"\n";
    }

    {
        timer.startAndPrint("marray");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += a(i);
        }
        timer.stopAndPrint();
        timer.reset();
        std::cout<<"s "<<s<<"\n";
    }

    
    {
        timer.startAndPrint("marrayview");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += v(i);
        }
        timer.stopAndPrint();
        timer.reset();
        std::cout<<"s "<<s<<"\n";
    }

   
    {
        auto p = &v(0);
        const auto vShape = vigra::Shape1(shape[0]);
        vigra::MultiArrayView<1,float> vv(vShape, p);
        
        timer.startAndPrint("vigra");
        timer.reset();
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            s += vv(i);
        }
        timer.stopAndPrint();
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
        timer.stopAndPrint();
        timer.reset();
        std::cout<<p[0]<<"\n";
    }

    {
    
        timer.startAndPrint("marray");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            a(i) = s++;
        }
        timer.stopAndPrint();
        timer.reset();
        std::cout<<a(0)<<"\n";
    }

    
    {
    
        timer.startAndPrint("marrayview");
        float s = 0;
        for(auto i=0; i<shape[0]; ++i){
            v(i) = s++;
        }
        timer.stopAndPrint();
        timer.reset();
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
        timer.stopAndPrint();
        timer.reset();
        std::cout<<vv(0)<<"\n";
    }
}



int main( int argc , char *argv[] ){

    //std::cout << "1D-Write benchmark:" << std::endl;
    //bench1dWrite(50);

    std::cout << std::endl;
    std::cout << "Thorstens 1D - benchmark" << std::endl;
    benchThorsten();
}
