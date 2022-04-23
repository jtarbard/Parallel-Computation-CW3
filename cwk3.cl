// Kernel for the heat equation.
__kernel
void test(__constant float *hostGrid, __global float *newGrid)
{
    int gid = get_global_id(0);

    newGrid[gid] = hostGrid[gid] * 2;
}