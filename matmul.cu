/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#import <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <omp.h>

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

#define BLOCK_SIZE 16

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float

void init(int M, int N, REAL * A) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

double maxerror(int M, int N, REAL * A, REAL *B) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i*N+j] - B[i*N+j]) / A[i*N+j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed_base, elapsed_openmp, elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*4); /* we use 5 matrix in this example */
    /* below is a cast from memory buffer to a 2-d row-major array */
    REAL *A = heap_buffer;
    REAL *B = &heap_buffer[N*N];
    REAL *C_base = &heap_buffer[2*N*N];
    REAL *C_openmp = &heap_buffer[3*N*N];

    srand48((1 << 12));
    init(N, N, A);
    init(N, N, B);

    

    /* example run */
    elapsed_base = read_timer();
    matmul_base(N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);

    elapsed_openmp = read_timer();
    matmul_openmp(N, A, B, C_openmp, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    /* call and timing for the three CUDA versions */
    /* there are three devices you can use on gpu.secs.oakland.edu, 0, 2, 3. 
     * 1 is a graphical card with less computation capability.
     */
    cudaSetDevice(0); 
    //call and time for matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
    elapsed_cuda_v1 = read_timer();
    matmul_cuda_v1_vanilla(N, A, B, C_base);
    elapsed_cuda_v1 = (read_timer() - elapsed_cuda_v1);
    //call and time for matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);
    elapsed_cuda_v2 = read_timer();
    matmul_cuda_v2_shmem(N, A, B, C_base);
    elapsed_cuda_v2 = (read_timer() - elapsed_cuda_v2);
    //call and time for matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);

    printf("======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));
    printf("matmul_cuda_v1:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v1 * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v1)), maxerror(N, N, C_base, C_openmp));
    printf("matmul_cuda_v1:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v2 * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v2)), maxerror(N, N, C_base, C_openmp));

    /* put other printf statements for outputing results for GPU execution */
    free(heap_buffer);
    return 0;
}

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int N)
{
    Matrix subMatrix;
    subMatrix.width = BLOCK_SIZE;
    subMatrix.height = BLOCK_SIZE;
    subMatrix.stride = N;
    subMatrix.elements = &A.elements[N * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return subMatrix;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(N,A,B,C,num_tasks) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

__global__ void global_kernel(int N, REAL *A, REAL *B, REAL *C)
{
    float cValue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < N; ++e) cValue += A[row * N + e] * B[e * N + col];
    C[row * N + col] = cValue;
}

__global__ void shared_kernel(int N, Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol, N);

    float cValue = 0.0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (N/ BLOCK_SIZE); ++m) {

        Matrix Asub = GetSubMatrix(A, blockRow, m, N);

        Matrix Bsub = GetSubMatrix(B, m, blockCol, N);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        
        __syncthreads();
        printf("matmul_cuda_v1:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v1 * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v1)), maxerror(N, N, C_base, C_openmp));

        for (int e = 0; e < BLOCK_SIZE; ++e) cValue += As[row][e] * Bs[e][col];
        __syncthreads();
    }

    SetElement(Csub, row, col, cValue);
}

/*
 * call to kernel that uses GPU global memory
 */
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C) {
    size_t size = N * N *sizeof(REAL);

    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

    global_kernel<<<dimGrid, dimBlock>>>(N, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/*
 * call to kernel that use GPU shared memory
 */
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C) 
{
    Matrix d_A;
    d_A.width = N;
    d_A.stride = N; 
    d_A.height = N;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = N;
    d_B.stride = N; 
    d_B.height = N;
    size = N * N * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = N;
    d_C.stride = N;
    d_C.height = N;
    size = N * N * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    shared_kernel<<<dimGrid, dimBlock>>>(N, d_A, d_B, d_C);

    cudaMemcpy(C, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

/*
 * call to sgemm of cublas library 
 */
void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C) {

}
