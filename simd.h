#ifndef _SIMD_H_
#define _SIMD_H_

#ifdef __AVX__

#include <immintrin.h>

typedef __m256 fvec;
typedef __m256d dvec;

#define FVEC_SIZE 8
#define DVEC_SIZE 4

// Perform an SIMD add of the two given vectors
__attribute__((always_inline)) inline fvec vec_add(fvec a, fvec b) {
    return _mm256_add_ps(a, b);
}
__attribute__((always_inline)) inline dvec vec_add(dvec a, dvec b) {
    return _mm256_add_pd(a, b);
}

// Perform an SIMD subtraction of the two given vectors
__attribute__((always_inline)) inline fvec vec_sub(fvec a, fvec b) {
    return _mm256_sub_ps(a, b);
}
__attribute__((always_inline)) inline dvec vec_sub(dvec a, dvec b) {
    return _mm256_sub_pd(a, b);
}

// Perform an SIMD multiplication of the two given vectors
__attribute__((always_inline)) inline fvec vec_mul(fvec a, fvec b) {
    return _mm256_mul_ps(a, b);
}
__attribute__((always_inline)) inline dvec vec_mul(dvec a, dvec b) {
    return _mm256_mul_pd(a, b);
}

// Perform an SIMD division of the two given vectors
__attribute__((always_inline)) inline fvec vec_div(fvec a, fvec b) {
    return _mm256_div_ps(a, b);
}
__attribute__((always_inline)) inline dvec vec_div(dvec a, dvec b) {
    return _mm256_div_pd(a, b);
}


// Return a 256-bit vector with all elements set to zero
__attribute__((always_inline)) inline fvec fvec_setzero() {
    return _mm256_setzero_ps();
}
__attribute__((always_inline)) inline dvec dvec_setzero() {
    return _mm256_setzero_pd();
}

// Return a 256-bit vector with all elements intialized to specified scalar
__attribute__((always_inline)) inline fvec fvec_set1(float x) {
    return _mm256_set1_ps(x);
}
__attribute__((always_inline)) inline dvec dvec_set1(double x) {
    return _mm256_set1_pd(x);
}

// Read a vector from the given address
__attribute__((always_inline)) inline fvec vec_load(float const *mem_addr) {
    return _mm256_loadu_ps(mem_addr);
}
__attribute__((always_inline)) inline dvec vec_load(double const *mem_addr) {
    return _mm256_loadu_pd(mem_addr);
}

// Store a vector to the given address
__attribute__((always_inline)) inline void vec_store(float *mem_addr, fvec a) {
    _mm256_storeu_ps(mem_addr, a);
}
__attribute__((always_inline)) inline void vec_store(double *mem_addr, dvec a) {
    _mm256_storeu_pd(mem_addr, a);
}

#endif

#endif
