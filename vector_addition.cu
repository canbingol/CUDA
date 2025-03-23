# include <iostream>
__global__ void vector_add(const float* A, const float* B,  float* C, int N){
    int i = blockIdx.x * blockDim.x +  threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    const int N = 10;
    float A[N], B[N], C[N];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    cudaMemcpy(d_a, A,N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B,N*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = ceil(N/block_size);
    vector_add<<<grid_size, block_size >>>(d_a,d_b,d_c,N);

    cudaMemcpy(C, d_c, N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}