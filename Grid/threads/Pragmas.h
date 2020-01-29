/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/Threads.h

    Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>
Author: Gianluca Filaci <g.filaci@ed.ac.uk>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#pragma once

#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)>(y)?(y):(x))
#endif

#define strong_inline     __attribute__((always_inline)) inline
#define UNROLL  _Pragma("unroll")

#if defined(__CUDA_ARCH__) || defined(__SYCL_DEVICE_ONLY__)
#define __GRID_DEVICE_ONLY__
#endif

//////////////////////////////////////////////////////////////////////////////////
// New primitives; explicit host thread calls, and accelerator data parallel calls
//////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#define GRID_OMP
#include <omp.h>
#endif

#ifdef GRID_OMP
#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define thread_num(a) omp_get_thread_num()
#define thread_max(a) omp_get_max_threads()
#else 
#define DO_PRAGMA_(x) 
#define DO_PRAGMA(x) 
#define thread_num(a) (0)
#define thread_max(a) (1)
#endif

#ifdef CL_SYCL_LANGUAGE_VERSION
#define GRID_SYCL
#include <CL/sycl.hpp>
#include <Grid/threads/AccessorContainer.h>
extern AccessorContainer accessor_container;
#define require_host(dummy)        accessor_container.create_host_accessors();
#define require_device(cgh)        accessor_container.require(cgh);
#define push_accessor_to_list(acc) accessor_container.push_back(acc);
#define delete_accessors(dummy)    accessor_container.delete_host_accessors(); accessor_container.clear();
#define clear_accessor_list(dummy) accessor_container.clear();
#else
#define require_host(dummy)
#define require_device(dummy)
#define push_accessor_to_list(dummy)
#define delete_accessors(dummy)
#define clear_accessor_list(dummy)
#endif

#define thread_for( i, num, ... )                         require_host(); DO_PRAGMA(omp parallel for schedule(static)) for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ; delete_accessors();
#define thread_foreach( i, container, ... )               require_host(); DO_PRAGMA(omp parallel for schedule(static)) for ( uint64_t i=container.begin();i<container.end();i++) { __VA_ARGS__ } ; delete_accessors();
#define thread_for_in_region( i, num, ... )               require_host(); DO_PRAGMA(omp for schedule(static))          for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ; delete_accessors();
#define thread_for_collapse2( i, num, ... )               require_host(); DO_PRAGMA(omp parallel for collapse(2))      for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ; delete_accessors();
#define thread_for_collapse( N , i, num, ... )            require_host(); DO_PRAGMA(omp parallel for collapse ( N ) )  for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ; delete_accessors();
#define thread_for_collapse_in_region( N , i, num, ... )  require_host(); DO_PRAGMA(omp for collapse ( N ))            for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ; delete_accessors();
#define thread_region                                                             DO_PRAGMA(omp parallel)
#define thread_critical                                                           DO_PRAGMA(omp critical)


//////////////////////////////////////////////////////////////////////////////////
// Accelerator primitives; fall back to threading
//////////////////////////////////////////////////////////////////////////////////
#ifdef __NVCC__
#define GRID_NVCC
#endif

#ifdef GRID_NVCC

#define GRID_LANE_IDX      threadIdx.y
#define syncSIMT(dummy)  __syncwarp();

extern uint32_t gpu_threads;

#define accelerator        __host__ __device__
#define accelerator_inline __host__ __device__ inline

template<typename lambda>  __global__
void LambdaApplySIMT(uint64_t Isites, uint64_t Osites, lambda Lambda)
{
  uint64_t isite = threadIdx.y;
  uint64_t osite = threadIdx.x+blockDim.x*blockIdx.x;
  if ( (osite <Osites) && (isite<Isites) ) {
    Lambda(isite,osite);
  }
}

/////////////////////////////////////////////////////////////////
// Internal only really... but need to call when 
/////////////////////////////////////////////////////////////////
#define accelerator_barrier(dummy)				\
  {								\
    cudaDeviceSynchronize();					\
    cudaError err = cudaGetLastError();				\
    if ( cudaSuccess != err ) {					\
      printf("Cuda error %s \n", cudaGetErrorString( err )); \
      puts(__FILE__); \
      printf("Line %d\n",__LINE__);					\
      exit(0);							\
    }								\
  }

// Copy the for_each_n style ; Non-blocking variant
#define accelerator_forNB( iterator, num, nsimd, ... )			\
  {									\
    typedef uint64_t Iterator;						\
    auto lambda = [=] accelerator (Iterator lane,Iterator iterator) mutable { \
      __VA_ARGS__;							\
    };									\
    dim3 cu_threads(gpu_threads,nsimd);					\
    dim3 cu_blocks ((num+gpu_threads-1)/gpu_threads);			\
    LambdaApplySIMT<<<cu_blocks,cu_threads>>>(nsimd,num,lambda);	\
  }

// Copy the for_each_n style ; Non-blocking variant (default
#define accelerator_for( iterator, num, nsimd, ... )		\
  accelerator_forNB(iterator, num, nsimd, { __VA_ARGS__ } );	\
  accelerator_barrier(dummy);

#elif defined(GRID_SYCL)

#define GRID_LANE_IDX __spirv_BuiltInGlobalInvocationId.x
// FIXME: can try SPIR-V commands from intel::sub_group barrier, but
// on SIMD hardware a sub-group barrier is expected to be a no-op
#define syncSIMT(dummy)

extern std::unique_ptr<cl::sycl::queue> queue;

#define accelerator
#define accelerator_inline inline

#define accelerator_barrier(dummy)				\
  {								\
    try {							\
      queue->wait_and_throw();					\
    } catch (cl::sycl::exception const& e) {			\
      std::cout << "Caught synchronous SYCL exception:\n"	\
		<< e.what() << std::endl;			\
      exit(0);							\
    }								\
    clear_accessor_list();					\
  }

#define accelerator_forNB( iterator, num, nsimd, ... )			\
  {									\
    queue->submit([&] (cl::sycl::handler &cgh) {			\
	require_device(cgh);						\
	cgh.parallel_for(cl::sycl::range<2>(nsimd,num),			\
			 [=] (cl::sycl::id<2> id) mutable {		\
			   auto iterator = id.get(1);			\
			   __VA_ARGS__;					\
			 });						\
      });                                                               \
  }

#define accelerator_for( iterator, num, nsimd, ... )		\
  accelerator_forNB(iterator, num, nsimd, { __VA_ARGS__ } );	\
  accelerator_barrier();

#else

#define GRID_LANE_IDX

#define accelerator 
#define accelerator_inline strong_inline
#define accelerator_for(iterator,num,nsimd, ... )   thread_for(iterator, num, { __VA_ARGS__ });
#define accelerator_forNB(iterator,num,nsimd, ... ) thread_for(iterator, num, { __VA_ARGS__ });
#define accelerator_barrier(dummy) 

#endif
