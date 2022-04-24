// Kernel for the heat equation.
__kernel
void heat(__constant int *Np, __constant float *hostGrid, __global float *newGrid)
{

    int i = get_global_id(0); 
    int N = *Np;

    if( i < N || i > (N*(N-1)-1) || i % N == 0 || (i+1) % N == 0)
    {
        newGrid[i] = 0.0;
    }
    else
    {
        newGrid[i] = 0.25 * (hostGrid[i-N] + hostGrid[i+N] + hostGrid[i-1] + hostGrid[i+1]);
    }
}