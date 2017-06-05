#pragma once

#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>




/** \def NIFTY_CHECK_OP(a,op,b,message)
    \brief macro for runtime checks
    
    \warning The check is done 
        <B> even in Release mode </B> 
        (therefore if NDEBUG <B>is</B> defined)

    \param a : first argument (like a number )
    \param op : operator (== )
    \param b : second argument (like a number )
    \param message : error message (as "my error")

    <b>Usage:</b>
    \code
        int a = 1;
        NIFTY_CHECK_OP(a, ==, 1, "this should never fail")
        NIFTY_CHECK_OP(a, >=, 2, "this should fail")
    \endcode
*/
#define NIFTY_CHECK_OP(a,op,b,message) \
    if(!  static_cast<bool>( a op b )   ) { \
       std::stringstream s; \
       s << "Nifty Error: "<< message <<"\n";\
       s << "Nifty check :  " << #a <<#op <<#b<< "  failed:\n"; \
       s << #a " = "<<a<<"\n"; \
       s << #b " = "<<b<<"\n"; \
       s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
       throw std::runtime_error(s.str()); \
    }

/** \def NIFTY_CHECK(expression,message)
    \brief macro for runtime checks
    
    \warning The check is done 
        <B> even in Release mode </B> 
        (therefore if NDEBUG <B>is</B> defined)

    \param expression : expression which can evaluate to bool
    \param message : error message (as "my error")

    <b>Usage:</b>
    \code
        int a = 1;
        NIFTY_CHECK_OP(a==1, "this should never fail")
        NIFTY_CHECK_OP(a>=2, "this should fail")
    \endcode
*/
#define NIFTY_CHECK(expression, message) if(!(expression)) { \
   std::stringstream s; \
   s << message <<"\n";\
   s << "Nifty assertion " << #expression \
   << " failed in file " << __FILE__ \
   << ", line " << __LINE__ << std::endl; \
   throw std::runtime_error(s.str()); \
 }


#define NIFTY_TEST(expression) NIFTY_CHECK(expression,"")

#define NIFTY_TEST_OP(a,op,b) NIFTY_CHECK_OP(a,op,b,"")

#define NIFTY_CHECK_EQ_TOL(a,b,tol) \
    if( std::abs(a-b) > tol){ \
        std::stringstream s; \
        s<<"Nifty assertion "; \
        s<<"\""; \
        s<<" | "<< #a <<" - "<<#b<<"| < " #tol <<"\" "; \
        s<<"  failed with:\n"; \
        s<<#a<<" = "<<a<<"\n";\
        s<<#b<<" = "<<b<<"\n";\
        s<<#tol<<" = "<<tol<<"\n";\
        throw std::runtime_error(s.str()); \
    }

#define NIFTY_TEST_EQ_TOL(a,b,tol) NIFTY_CHECK_EQ_TOL(a,b,tol)

#define NIFTY_CHECK_NUMBER(number) \
   { \
   std::stringstream s; \
   s << "Nifty assertion failed in file " << __FILE__ \
   << ", line " << __LINE__ << std::endl; \
    if(std::isnan(number))\
        throw std::runtime_error(s.str()+" number is nan"); \
    if(std::isinf(number))\
        throw std::runtime_error(s.str()+"number is inf");\
    }

#ifdef NDEBUG
    #ifdef NIFTY_DEBUG 
        #define NIFTY_DO_DEBUG
    #endif
#else
    #ifdef NIFTY_DEBUG 
        #define NIFTY_DO_DEBUG
    #endif
#endif


/** \def NIFTY_ASSERT_OP(a,op,b,message)
    \brief macro for runtime checks
    
    \warning The check is <B>only</B> done in
        in Debug mode (therefore if NDEBUG is <B>not</B> defined)

    \param a : first argument (like a number )
    \param op : operator (== )
    \param b : second argument (like a number )
    \param message : error message (as "my error")

    <b>Usage:</b>
    \code
        int a = 1;
        NIFTY_ASSERT_OP(a, ==, 1) // will not fail here
        NIFTY_ASSERT_OP(a, >=, 2) // will fail here
    \endcode
*/
#ifdef NDEBUG
   #ifndef NIFTY_DEBUG 
      #define NIFTY_ASSERT_OP(a,op,b) { }
   #else
      #define NIFTY_ASSERT_OP(a,op,b) \
      if(!  static_cast<bool>( a op b )   ) { \
         std::stringstream s; \
         s << "Nifty assertion :  " << #a <<#op <<#b<< "  failed:\n"; \
         s << #a " = "<<a<<"\n"; \
         s << #b " = "<<b<<"\n"; \
         s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
         throw std::runtime_error(s.str()); \
      }
   #endif
#else
   #define NIFTY_ASSERT_OP(a,op,b) \
   if(!  static_cast<bool>( a op b )   ) { \
      std::stringstream s; \
      s << "Nifty assertion :  " << #a <<#op <<#b<< "  failed:\n"; \
      s << #a " = "<<a<<"\n"; \
      s << #b " = "<<b<<"\n"; \
      s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
      throw std::runtime_error(s.str()); \
   }
#endif

/** \def NIFTY_ASSERT(expression,message)
    \brief macro for runtime checks
    
    \warning The check is <B>only</B> done in
        in Debug mode (therefore if NDEBUG is <B>not</B> defined)

    \param expression : expression which can evaluate to bool

    <b>Usage:</b>
    \code
        int a = 1;
        NIFTY_ASSERT(a == 1) // will not fail here 
        NIFTY_ASSERT(a >= 2) // will fail here
    \endcode
*/
#ifdef NDEBUG
   #ifndef NIFTY_DEBUG
      #define NIFTY_ASSERT(expression) {}
   #else
      #define NIFTY_ASSERT(expression) if(!(expression)) { \
         std::stringstream s; \
         s << "Nifty assertion " << #expression \
         << " failed in file " << __FILE__ \
         << ", line " << __LINE__ << std::endl; \
         throw std::runtime_error(s.str()); \
      }
   #endif
#else
      #define NIFTY_ASSERT(expression) if(!(expression)) { \
         std::stringstream s; \
         s << "Nifty assertion " << #expression \
         << " failed in file " << __FILE__ \
         << ", line " << __LINE__ << std::endl; \
         throw std::runtime_error(s.str()); \
      }
#endif





