#include <bitset>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits.h>
#include <new>
#include <stdint.h>
#include <stdio.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif


void PrintBitArray(uint64_t * arr, unsigned len){
  for(unsigned i = 0; i < len; ++i)
    std::cout << std::bitset<64>(arr[i]) << std::endl;
}

void PrintBlock(uint64_t num){
  for(unsigned j = 0; j < 64; j += 8)
    std::cout << std::bitset<8>(char(num >> j)) << std::endl;
  std::cout << std::endl;
}

void EightByEightSubmatricesNaive(uint64_t * src, uint64_t * dest, unsigned len){
  // assumes dest is initialized to all zeros.
  for(uint64_t * srcPtr = src; srcPtr < src + len; ++srcPtr){
    unsigned srcInd = (srcPtr - src) % 8;
    uint64_t * destBlock = dest + 8*((srcPtr - src)/8);
    for(unsigned destInd = 0; destInd < 8; ++destInd){
      uint64_t * destPtr = destBlock + destInd;
      *destPtr |= ((*srcPtr >> 8*destInd) & 0xFF) << (8*srcInd);
    }
    // num at srcPtr contributes the srcInd byte in each submatrix.
    // destInd byte of that number belongs to submatrix at destPtr.
  }
}

void EightByEightSubmatricesAVX(uint64_t * src, uint64_t * dest, unsigned len){
  assert(len % 8 == 0);
#ifdef __AVX2__
  uint64_t * p = src;
  uint64_t * endp =  src+len;
  uint64_t * q = dest;
  do {
    // process 8 rows at a time, split into two __m256i's halves.
    // within each half, move lower 32 of each uint64
    // into bottom 128, upper 32 into top 128.
    __m256i gatherBy = _mm256_set_epi32(7,5,3,1,6,4,2,0);
    __m256i permuted1 = _mm256_i32gather_epi32 (p, gatherBy, 4);
    __m256i permuted2 = _mm256_i32gather_epi32 (p+32/sizeof(p[0]), gatherBy, 4);
    
    // bring together bytes in the same position.
    __m256i shuffleBy = _mm256_set_epi8(15,11,7,3,
                                         14,10,6,2,
                                         13, 9,5,1,
                                         12, 8,4,0,

                                         15,11,7,3,
                                         14,10,6,2,
                                         13, 9,5,1,
                                         12, 8,4,0);
    __m256i temp[2];
    temp[0] = _mm256_shuffle_epi8(permuted1, shuffleBy);
    temp[1] = _mm256_shuffle_epi8(permuted2, shuffleBy);
    // top halves of the 8 submatrices are now in temp[0],
    // bottom halves in temp[1].

    // bring together top and bottom halves.
    __m256i firstFourBlocks = _mm256_i32gather_epi32(temp,
              _mm256_set_epi32(11,3,10,2,9,1,8,0), 4);
    __m256i secondFourBlocks = _mm256_i32gather_epi32(temp,
              _mm256_set_epi32(15,7,14,6,13,5,12,4), 4);
    
    _mm256_storeu_si256((__m256i*)q, firstFourBlocks);
    _mm256_storeu_si256((__m256i*)(q+32/sizeof(q[0])), secondFourBlocks);
    p += 64/sizeof(p[0]);
    q += 64/sizeof(q[0]);

    // a combination of _mm256_permutevar8x32_epi32
    // and _mm256_shuffle_epi8 would also work,
    // but in this case a gather is simpler (and faster).
  } while (p < endp);
#else
  std::cout << "AVX2 not supported" << std::endl;
  exit(1);
#endif
}

int main(int, char *argv[]){
  static const bool DEBUG = false;
  static const unsigned M = 16;

  if (DEBUG){
    srand (time(NULL));
    uint64_t test[M] = {0};
    uint64_t avxResult[M] = {0};
    uint64_t naiveResult[M] = {0};
    for(unsigned i = 0; i < M; ++i)
      test[i] = uint64_t(rand()) | (uint64_t(rand()) << 32);
    EightByEightSubmatricesAVX(test, avxResult, M);
    EightByEightSubmatricesNaive(test, naiveResult, M);
    for (unsigned i = 0; i < M; ++i){
      if(avxResult[i] != naiveResult[i]){
        std::cout << "issue at submatrix " << i << std::endl;
        PrintBitArray(test, M);
        std::cout << "expected"  << std::endl;
        PrintBlock(naiveResult[i]);
        std::cout << "actual" << std::endl;
        PrintBlock(avxResult[i]);
        assert(false);
      }
    }
  }

  uint64_t demo[M];
  uint64_t submatrices[M] = {0};
  // hand-chosen numbers: enough variation that the submatrices are different,
  // but not so much that it's chaotic and random-looking.
  for (unsigned i = 0; i < 4; ++i)
    demo[i] = 0xFF0FFF0F00FF00FFULL;
  for (unsigned i = 4; i < 8; ++i)
    demo[i] = 0xFF00FF00F0FFF0FFULL;
  for (unsigned i = 8; i < 16; ++i)
    demo[i] = 0x00FF00FF00FF00FFULL << i;
  
  std::cout << "original bit array:" << std::endl;
  PrintBitArray(demo, M);

  std::cout << "8-by-8 submatrix blocks, right to left:" << std::endl;
  EightByEightSubmatricesAVX(demo, submatrices, M);
  for(unsigned i = 0; i < M; ++i)
    PrintBlock(submatrices[i]);

#define N 100000
  assert(8*N < UINT_MAX);
  uint64_t * src = new uint64_t[8*N];
  srand (time(NULL));
  for(unsigned i = 0; i < 8*N; ++i)
    src[i] = uint64_t(rand()) | (uint64_t(rand()) << 32);
  std::cout << "time to apply to matrix of " << 8*N << " rows" << std::endl;
  for(unsigned n = 0; n < 2; ++n){
    uint64_t * dest = new uint64_t[8*N]();
    auto startTime = std::chrono::high_resolution_clock::now();
    n == 0 ? EightByEightSubmatricesNaive(src, dest, N) :
            EightByEightSubmatricesAVX(src, dest, N);
    auto endTime = std::chrono::high_resolution_clock::now();
    n == 0 ? std::cout << "without AVX: " : std::cout << "with AVX: ";
    std::chrono::duration<double, std::milli> passed = endTime - startTime;
    std::cout << passed.count() << " ms " << std::endl;
    delete[] dest;
  }
  delete[] src;
  return 0;
}