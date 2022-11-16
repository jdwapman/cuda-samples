/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of doubleing-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Matrix size */
// #define N (4096)

/* Host implementation of a simple version of dgemm */
static void simple_dgemm(int n, double alpha, const double *A, const double *B,
                         double beta, double *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      double prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

/* Main */
int main(int argc, char **argv) {
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  cublasStatus_t status;
  double *h_A;
  double *h_B;
  double *h_C;
  double *h_C_ref;
  double *d_A = 0;
  double *d_B = 0;
  double *d_C = 0;
  double alpha = 1.0f;
  double beta = 0.0f;
  // int n2 = N * N;
  int i;
  double error_norm;
  double ref_norm;
  double diff;
  cublasHandle_t handle;

  // int dev = findCudaDevice(argc, (const char **)argv);

  // if (dev == -1) {
  //   return EXIT_FAILURE;
  // }

  /* Initialize CUBLAS */
  printf("simpleCUBLAS test running..\n");

  std::vector<float> runtimes;

  int timing_iterations = 10;

  for (int i = 0; i < timing_iterations; i++) {
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! CUBLAS initialization error\n");
      return EXIT_FAILURE;
    }

    /* Allocate host memory for the matrices */
    h_A = reinterpret_cast<double *>(malloc(M * K * sizeof(h_A[0])));

    if (h_A == 0) {
      fprintf(stderr, "!!!! host memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    h_B = reinterpret_cast<double *>(malloc(K * N * sizeof(h_B[0])));

    if (h_B == 0) {
      fprintf(stderr, "!!!! host memory allocation error (B)\n");
      return EXIT_FAILURE;
    }

    h_C = reinterpret_cast<double *>(malloc(M * N * sizeof(h_C[0])));

    if (h_C == 0) {
      fprintf(stderr, "!!!! host memory allocation error (C)\n");
      return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < M * K; i++) {
      h_A[i] = rand() / static_cast<double>(RAND_MAX);
    }

    for (i = 0; i < K * N; i++) {
      h_B[i] = rand() / static_cast<double>(RAND_MAX);
    }

    for (i = 0; i < M * N; i++) {
      h_C[i] = rand() / static_cast<double>(RAND_MAX);
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc(reinterpret_cast<void **>(&d_A), M * K * sizeof(d_A[0])) !=
        cudaSuccess) {
      fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
      return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_B), K * N * sizeof(d_B[0])) !=
        cudaSuccess) {
      fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
      return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_C), M * N * sizeof(d_C[0])) !=
        cudaSuccess) {
      fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
      return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(M * K, sizeof(h_A[0]), h_A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! device access error (write A)\n");
      return EXIT_FAILURE;
    }

    status = cublasSetVector(K * N, sizeof(h_B[0]), h_B, 1, d_B, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! device access error (write B)\n");
      return EXIT_FAILURE;
    }

    status = cublasSetVector(M * N, sizeof(h_C[0]), h_C, 1, d_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! device access error (write C)\n");
      return EXIT_FAILURE;
    }

    /* Performs operation using plain C code */
    // simple_dgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    /* Performs operation using cublas */
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A,
                         M, d_B, K, &beta, d_C, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed Time: %f ms\n", milliseconds);
    runtimes.push_back(milliseconds);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }

    /* Allocate host memory for reading back the result from device memory */
    h_C = reinterpret_cast<double *>(malloc(M * N * sizeof(h_C[0])));

    if (h_C == 0) {
      fprintf(stderr, "!!!! host memory allocation error (C)\n");
      return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(M * N, sizeof(h_C[0]), d_C, 1, h_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! device access error (read C)\n");
      return EXIT_FAILURE;
    }

    /* Check result against reference */
    // error_norm = 0;
    // ref_norm = 0;

    // for (i = 0; i < n2; ++i) {
    //   diff = h_C_ref[i] - h_C[i];
    //   error_norm += diff * diff;
    //   ref_norm += h_C_ref[i] * h_C_ref[i];
    // }

    // error_norm = static_cast<double>(sqrt(static_cast<double>(error_norm)));
    // ref_norm = static_cast<double>(sqrt(static_cast<double>(ref_norm)));

    // if (fabs(ref_norm) < 1e-7) {
    //   fprintf(stderr, "!!!! reference norm is 0\n");
    //   return EXIT_FAILURE;
    // }

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    if (cudaFree(d_A) != cudaSuccess) {
      fprintf(stderr, "!!!! memory free error (A)\n");
      return EXIT_FAILURE;
    }

    if (cudaFree(d_B) != cudaSuccess) {
      fprintf(stderr, "!!!! memory free error (B)\n");
      return EXIT_FAILURE;
    }

    if (cudaFree(d_C) != cudaSuccess) {
      fprintf(stderr, "!!!! memory free error (C)\n");
      return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    // if (error_norm / ref_norm < 1e-6f) {
    //   printf("simpleCUBLAS test passed.\n");
    //   exit(EXIT_SUCCESS);
    // } else {
    //   printf("simpleCUBLAS test failed.\n");
    //   exit(EXIT_FAILURE);
    // }
  }

  float average_runtime = 0;

  for (int i = 0; i < runtimes.size(); i++) {
    average_runtime += runtimes[i];
  }

  average_runtime /= runtimes.size();

  // Calculate double precision TFLOPS from milliseconds for matrix
  // multiplication
  double flops = 2.0 * M * N * K;
  double tflops = (flops * 1.0e-12) / (average_runtime / 1000.0);
  printf("TFLOPS: %f\n", tflops);
}
