// k. j. roche
// simple C++ gradient computation to exhibit use of FFT on regularized lattice
// here I use FFTW and CUDA FFT (CUFFT)

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <algorithm>
#include <iomanip> // Include the <iomanip> header for setprecision

// from the C standard
#include <cstdlib>
#include <cmath>
#include <fftw3.h>

// add a timer for performance analysis
#include <chrono>

// add a cuda section for the plane wave just to exhibit the beauty of it all ...
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void kr_fast_scale_copy(cufftDoubleComplex *in, cufftDoubleComplex *out, int n, double scale_factor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
    {
        out[idx].x = scale_factor * in[idx].x;
        out[idx].y = scale_factor * in[idx].y;
    }
}

__global__ void kr_fast_scale(cufftDoubleComplex *in, int n, double *scale_factor)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
    {
        in[idx].x *= scale_factor[idx];
        in[idx].y *= scale_factor[idx];
    }
}

void measure_gradient_performance(int n, int ntrials, std::ofstream& results)
{
    double avg_gpu_time_ms = 0.0;
    double avg_cpu_time_s = 0.0;

    // default lattice
    int nx = n;
    int ny = n;
    int nz = n;
    ny = nz = nx;
    std::cout << "Nx,Ny,Nz:" << nx << "," << ny << "," << nz << std::endl;

    // theoretical complexity
    double complexity = 3 * nx * nx * nx * log(nx);
    std::cout << "F[f]\t\t complexity for (" << nx << ")^3 = " << complexity << std::endl;
    complexity *= 8.;
    complexity += (9 * nx * nx * nx);
    std::cout << "gradient[f]\t complexity for (" << nx << ")^3 = " << complexity << std::endl;
    results << nx << ", " << complexity << ", ";
    // integral number of points in the volume
    int nxyz = nx * ny * nz;

    const double PI = 3.14159265359;

    for(int k=0; k<ntrials; k++)
    {
        // why not just work in L=1 where D = [a,b] and L = b-a
        const double lx = 1.0;
        const double ly = 1.0;
        const double lz = 1.0;

        // clearly spatial increment
        const double dx = lx / nx;
        const double dy = ly / ny;
        const double dz = lz / nz;

        std::vector<double> xa(nx * ny * nz);
        std::vector<double> ya(nx * ny * nz);
        std::vector<double> za(nx * ny * nz);

        /* comment out below if not wanting 0,0,0 in the corner */
        /*
            for (int ix = 0; ix < nx; ix++)
            {
                xx[ix] = dx * static_cast<double>(ix);
            }

            for (int iy = 0; iy < ny; iy++)
            {
                yy[iy] = dy * static_cast<double>(iy);
            }

            for (int iz = 0; iz < nz; iz++)
            {
                zz[iz] = dz * static_cast<double>(iz);
            }

            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    for (int iz = 0; iz < nz; iz++)
                    {
                        int index = iz + nz * (iy + ny * ix);
                        xa[index] = xx[ix];
                        ya[index] = yy[iy];
                        za[index] = zz[iz];
                    }
                }
            }
        */

        // Define the center of the lattice to be 0,0,0
        double center_x = static_cast<double>(nx) / 2.0;
        double center_y = static_cast<double>(ny) / 2.0;
        double center_z = static_cast<double>(nz) / 2.0;

        { // open code block to init so that xx,yy,zz are removed from memory out of scope
            std::vector<double> xx(nx);
            std::vector<double> yy(ny);
            std::vector<double> zz(nz);

            for (int ix = 0; ix < nx; ix++)
            {
                xx[ix] = dx * static_cast<double>(ix) - center_x * dx;
            }

            for (int iy = 0; iy < ny; iy++)
            {
                yy[iy] = dy * static_cast<double>(iy) - center_y * dy;
            }

            for (int iz = 0; iz < nz; iz++)
            {
                zz[iz] = dz * static_cast<double>(iz) - center_z * dz;
            }

            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    for (int iz = 0; iz < nz; iz++)
                    {
                        int index = iz + nz * (iy + ny * ix);
                        xa[index] = xx[ix];
                        ya[index] = yy[iy];
                        za[index] = zz[iz];
                    }
                }
            }
        }

        // KR:: note - to recover ix,iy,iz from index
        // int ix = index % nx; int iy = (index / nx) % ny; int iz = index / (nx * ny);

        // gpu data structures for coordinate space
        cufftDoubleReal *g_xyz;
        cudaError_t icudaError = cudaMalloc((void **)&g_xyz, 3 * nxyz * sizeof(cufftDoubleReal));
        cudaMemcpy(g_xyz, xa.data(), nxyz * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
        cudaMemcpy(g_xyz + nxyz, ya.data(), nxyz * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
        cudaMemcpy(g_xyz + 2 * nxyz, za.data(), nxyz * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

        // see fftw manual for why this strange ordering
        // for instance ....
        // kkx[0] = 0.0π
        // kkx[1] = 0.125π
        // kkx[2] = 0.25π
        // kkx[3] = 0.375π
        // kkx[4] = 0.5π
        // kkx[5] = 0.625π
        // kkx[6] = 0.75π
        // kkx[7] = 0.875π
        // kkx[8] = -1.0π
        // kkx[9] = -0.875π
        // kkx[10] = -0.75π
        // kkx[11] = -0.625π
        // kkx[12] = -0.5π
        // kkx[13] = -0.375π
        // kkx[14] = -0.25π
        // kkx[15] = -0.125π

        std::vector<double> kx(nx * ny * nz);
        std::vector<double> ky(nx * ny * nz);
        std::vector<double> kz(nx * ny * nz);
        { // again open code block since I want kkx,kky,kkz gone after initializing the lattice
            std::vector<double> kkx(nx);
            std::vector<double> kky(ny);
            std::vector<double> kkz(nz);

            int i;
            kkx[0] = 0.;
            kky[0] = 0.;
            kkz[0] = 0.;

            for (i = 1; i <= nx / 2 - 1; i++)
            {
                kkx[i] = 2.0 * PI / lx * static_cast<double>(i);
            }
            int j = -i;
            for (i = nx / 2; i < nx; i++)
            {
                kkx[i] = 2.0 * PI / lx * static_cast<double>(j);
                j++;
            }
#ifdef VERBOSE
            if (nx <= 16)
                for (auto &k : kkx)
                    std::cout << "kkx: " << k << std::endl;
#endif
            for (i = 1; i <= ny / 2 - 1; i++)
            {
                kky[i] = 2.0 * PI / ly * static_cast<double>(i);
            }
            j = -i;
            for (i = ny / 2; i < ny; i++)
            {
                kky[i] = 2.0 * PI / ly * static_cast<double>(j);
                j++;
            }

            for (i = 1; i <= nz / 2 - 1; i++)
            {
                kkz[i] = 2.0 * PI / lz * static_cast<double>(i);
            }
            j = -i;
            for (i = nz / 2; i < nz; i++)
            {
                kkz[i] = 2.0 * PI / lz * static_cast<double>(j);
                j++;
            }

            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    for (int iz = 0; iz < nz; iz++)
                    {
                        int index = iz + nz * (iy + ny * ix);
                        kx[index] = kkx[ix];
                        ky[index] = kky[iy];
                        kz[index] = kkz[iz];
                    }
                }
            }
        }

        // gpu data structures for momentum space
        cufftDoubleReal *g_kxyz;                                                               // gpu k-space lattice
        icudaError = cudaMalloc((void **)&g_kxyz, 3 * nxyz * sizeof(cufftDoubleReal));         // get gpu memory
        cudaMemcpy(g_kxyz, kx.data(), nxyz * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice); // copy the k-space lattice to the gpu
        cudaMemcpy(g_kxyz + nxyz, ky.data(), nxyz * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
        cudaMemcpy(g_kxyz + 2 * nxyz, kz.data(), nxyz * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

        std::vector<std::complex<double>> wave(nxyz);
        std::vector<std::complex<double>> fft_3(nxyz);
        std::vector<std::complex<double>> d_dx(nxyz);
        std::vector<std::complex<double>> d_dy(nxyz);
        std::vector<std::complex<double>> d_dz(nxyz);
        std::vector<std::complex<double>> d_tmp(nxyz);

        // the equivalent gpu data structures
        cufftDoubleComplex *g_fft3, *g_wfft; // needed data structures
        icudaError = cudaMalloc((void **)&g_fft3, sizeof(cufftDoubleComplex) * nxyz);
        icudaError = cudaMalloc((void **)&g_wfft, sizeof(cufftDoubleComplex) * nxyz); // work array

        cufftDoubleComplex *g_ddx, *g_ddy, *g_ddz; // needed data structures for the gradient
        icudaError = cudaMalloc((void **)&g_ddx, sizeof(cufftDoubleComplex) * nxyz);
        icudaError = cudaMalloc((void **)&g_ddy, sizeof(cufftDoubleComplex) * nxyz);
        icudaError = cudaMalloc((void **)&g_ddz, sizeof(cufftDoubleComplex) * nxyz);

        // set up the discrete Fourier plans on the host and gpu
        // FFTW plans
        fftw_plan plan_f = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(&wave[0]), reinterpret_cast<fftw_complex *>(&fft_3[0]), FFTW_FORWARD, FFTW_MEASURE); // FFTW_ESTIMATE
        fftw_plan plan_b = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(&fft_3[0]), reinterpret_cast<fftw_complex *>(&d_tmp[0]), FFTW_BACKWARD, FFTW_MEASURE);

        // cuFFT plans
        cufftHandle g_plan_f, g_plan_b; // forward and backward gpu fft plans - these can be the same but ...
        cufftPlan3d(&g_plan_f, nx, ny, nz, CUFFT_Z2Z);
        cufftPlan3d(&g_plan_b, nx, ny, nz, CUFFT_Z2Z);

        // initialize a simple wavefunction for example ...
        std::cout << "Plane wave section:" << std::endl;
        for (int ix = 0; ix < nx; ix++)
        {
            for (int iy = 0; iy < ny; iy++)
            {
                for (int iz = 0; iz < nz; iz++)
                {
                    int index = iz + nz * (iy + ny * ix);
                    // wave[index] = std::exp(std::complex<double>(0.0, 1.0) * (xa[index] * kkx[ix] + ya[index] * kky[iy] + za[index] * kkz[iz]));
                    wave[index] = std::exp(std::complex<double>(0.0, 1.0) * (xa[index] * kx[index] + ya[index] * ky[index] + za[index] * kz[index]));
                }
            }
        }

        // Calculate the norm of the wavefunction
        double norm = 0.0;
        for (int j = 0; j < nxyz; j++)
        {
            norm += std::norm(wave[j]);
        }

        // Normalize the wavefunction
        double sqrtNorm = std::sqrt(norm);
        for (int j = 0; j < nxyz; j++)
        {
            wave[j] /= sqrtNorm;
        }

        // Initialize vectors for the analytical gradient, FFTW-computed gradient, and error
        std::vector<std::complex<double>> analytic_ddx(nxyz);
        std::vector<std::complex<double>> analytic_ddy(nxyz);
        std::vector<std::complex<double>> analytic_ddz(nxyz);

        for (int ix = 0; ix < nx; ix++)
        {
            for (int iy = 0; iy < ny; iy++)
            {
                for (int iz = 0; iz < nz; iz++)
                {
                    int index = iz + nz * (iy + ny * ix);
                    analytic_ddx[index] = std::complex<double>(0.0, 1.0) * kx[index] * wave[index] / static_cast<double>(nxyz);
                    analytic_ddy[index] = std::complex<double>(0.0, 1.0) * ky[index] * wave[index] / static_cast<double>(nxyz);
                    analytic_ddz[index] = std::complex<double>(0.0, 1.0) * kz[index] * wave[index] / static_cast<double>(nxyz);
                }
            }
        }

        // // first demonstrate the norm of the wavefunction
        // norm = 0.;
        // for (int i = 0; i < nxyz; i++)
        //     norm += std::norm(wave[i]);
        // std::cout << "the norm of the wave function is: " << sqrt(norm) << std::endl;

        // determine thread block layout
        // assume 512 threads per block for now - because we can!
        static int tpb = 512;
        int nblk = nxyz / tpb;
        if (nxyz % tpb != 0)
            nblk++;

        // copy the wavefunction to the GPU
        cudaMemcpy(g_fft3, wave.data(), nxyz * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice); // copy the (wave)function to the gpu

        //  Timer variables
        cudaEvent_t g_start, g_stop;
        // Start timer
        cudaEventCreate(&g_start);
        cudaEventCreate(&g_stop);

        // cheat and just report the gpu compute time
        cudaEventRecord(g_start, 0);

        // in-line my own GPU-based gradient computation
        if (cufftExecZ2Z(g_plan_f, g_fft3, g_fft3, CUFFT_FORWARD) != CUFFT_SUCCESS)
            std::cout << "error in cufftExecZ2Z FORWARD failed" << std::endl; // forward transform

        // gradX
        kr_fast_scale_copy<<<nblk, tpb>>>(g_fft3, g_wfft, nxyz, 1. / static_cast<double>(nxyz)); // threaded copy
        kr_fast_scale<<<nblk, tpb>>>(g_wfft, nxyz, g_kxyz);                                      // X: threaded scale (multiply)
        if (cufftExecZ2Z(g_plan_b, g_wfft, g_wfft, CUFFT_INVERSE) != CUFFT_SUCCESS)
            std::cout << "error: cufftExecZ2Z INVERSE failed" << std::endl;                     // reverse transform
        kr_fast_scale_copy<<<nblk, tpb>>>(g_wfft, g_ddx, nxyz, 1. / static_cast<double>(nxyz)); // d_dx in GPU memory

        // gradY
        kr_fast_scale_copy<<<nblk, tpb>>>(g_fft3, g_wfft, nxyz, 1. / static_cast<double>(nxyz)); // threaded copy
        kr_fast_scale<<<nblk, tpb>>>(g_wfft, nxyz, g_kxyz + nxyz);                               // Y: threaded scale (multiply)
        if (cufftExecZ2Z(g_plan_b, g_wfft, g_wfft, CUFFT_INVERSE) != CUFFT_SUCCESS)
            std::cout << "error: cufftExecZ2Z INVERSE failed" << std::endl;                     // reverse transform
        kr_fast_scale_copy<<<nblk, tpb>>>(g_wfft, g_ddy, nxyz, 1. / static_cast<double>(nxyz)); // d_dy in GPU memory

        // gradZ
        kr_fast_scale_copy<<<nblk, tpb>>>(g_fft3, g_wfft, nxyz, 1. / static_cast<double>(nxyz)); // threaded copy
        kr_fast_scale<<<nblk, tpb>>>(g_wfft, nxyz, g_kxyz + 2 * nxyz);                           // Z: threaded scale (multiply)
        if (cufftExecZ2Z(g_plan_b, g_wfft, g_wfft, CUFFT_INVERSE) != CUFFT_SUCCESS)
            std::cout << "error: cufftExecZ2Z INVERSE failed" << std::endl;                     // reverse transform
        kr_fast_scale_copy<<<nblk, tpb>>>(g_wfft, g_ddz, nxyz, 1. / static_cast<double>(nxyz)); // d_dz in GPU memory

        cudaDeviceSynchronize(); // Synchronize threads to ensure kernel is completed

        // Stop timer
        cudaEventRecord(g_stop, 0);
        cudaEventSynchronize(g_stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, g_start, g_stop);

        std::cout << "GPU Kernel execution time: " << elapsedTime << " ms" << std::endl;
        std::cout << "GPU Kernel FLOPs: " << (1000. * complexity / static_cast<double>(elapsedTime)) << std::endl;
        avg_gpu_time_ms += elapsedTime;

        // check the errors in the gpu computed gradient
        // first, copy the gradients from the GPU to the CPU
        cudaMemcpy(d_dx.data(), g_ddx, nxyz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_dy.data(), g_ddy, nxyz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_dz.data(), g_ddz, nxyz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

        // check the result compared to the expected analytical result
        std::vector<double> errorX(nxyz);
        std::vector<double> errorY(nxyz);
        std::vector<double> errorZ(nxyz);
        // Calculate the error between analytical and FFTW-computed gradients
        for (int i = 0; i < nxyz; i++)
        {
            errorX[i] = std::abs(analytic_ddx[i] - d_dx[i]);
            errorY[i] = std::abs(analytic_ddy[i] - d_dy[i]);
            errorZ[i] = std::abs(analytic_ddz[i] - d_dz[i]);
        }

        // Find the maximum error
        double maxErrorX = *std::max_element(errorX.begin(), errorX.end());
        double maxErrorY = *std::max_element(errorY.begin(), errorY.end());
        double maxErrorZ = *std::max_element(errorZ.begin(), errorZ.end());

        std::cout << "gpu: Max Error in Gradient (X): " << maxErrorX << std::endl;
        std::cout << "gpu: Max Error in Gradient (Y): " << maxErrorY << std::endl;
        std::cout << "gpu: Max Error in Gradient (Z): " << maxErrorZ << std::endl;

        // here I compute an error density out of curiosity
        //  Calculate the error density for each component (X, Y, Z)
        double errorDensityX = 0.0;
        double errorDensityY = 0.0;
        double errorDensityZ = 0.0;

        for (int i = 0; i < nxyz; i++)
        {
            errorDensityX += std::abs(analytic_ddx[i] - d_dx[i]);
            errorDensityY += std::abs(analytic_ddy[i] - d_dy[i]);
            errorDensityZ += std::abs(analytic_ddz[i] - d_dz[i]);
        }

        // Divide the sum of errors by the total number of grid points
        errorDensityX /= static_cast<double>(nxyz);
        errorDensityY /= static_cast<double>(nxyz);
        errorDensityZ /= static_cast<double>(nxyz);

        std::cout << "gpu: Error Density (X): " << errorDensityX << std::endl;
        std::cout << "gpu: Error Density (Y): " << errorDensityY << std::endl;
        std::cout << "gpu: Error Density (Z): " << errorDensityZ << std::endl;
        // END GPU CHECK

        // Compute spatial gradients on CPU
        // Start the timer
        auto start = std::chrono::high_resolution_clock::now();

        // Perform forward DFT
        fftw_execute(plan_f); // wave --> fft_3

        for (int j = 0; j < nxyz; j++)
        {
            wave[j] = fft_3[j] / static_cast<double>(nxyz);

            // d_dx
            fft_3[j] = wave[j] * std::complex<double>(0.0, kx[j]);
        }
        fftw_execute(plan_b); // fft_3 --> d_tmp
        for (int j = 0; j < nxyz; j++)
        {
            d_dx[j] = d_tmp[j] / static_cast<double>(nxyz);
        }

        for (int j = 0; j < nxyz; j++)
        {
            // d_dy
            fft_3[j] = wave[j] * std::complex<double>(0.0, ky[j]);
        }
        fftw_execute(plan_b);
        for (int j = 0; j < nxyz; j++)
        {
            d_dy[j] = d_tmp[j] / static_cast<double>(nxyz);
        }

        for (int j = 0; j < nxyz; j++)
        {
            // d_dz
            fft_3[j] = wave[j] * std::complex<double>(0.0, kz[j]);
        }
        fftw_execute(plan_b);

        for (int j = 0; j < nxyz; j++)
        {
            d_dz[j] = d_tmp[j] / static_cast<double>(nxyz);
        }

        // Calculate the execution time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Output the execution time and performance
        std::cout << "plane wave gradient - FFTW execution time: " << duration.count() << " seconds" << std::endl;
        std::cout << "plane wave gradient - FFTW FLOPs: " << complexity / static_cast<double>(duration.count()) << std::endl;
        avg_cpu_time_s += duration.count();

        // check the CPU result compared to the expected analytical result
        // Calculate the error between analytical and FFTW-computed gradients
        for (int i = 0; i < nxyz; i++)
        {
            errorX[i] = std::abs(analytic_ddx[i] - d_dx[i]);
            errorY[i] = std::abs(analytic_ddy[i] - d_dy[i]);
            errorZ[i] = std::abs(analytic_ddz[i] - d_dz[i]);
        }

        // Find the maximum error
        maxErrorX = *std::max_element(errorX.begin(), errorX.end());
        maxErrorY = *std::max_element(errorY.begin(), errorY.end());
        maxErrorZ = *std::max_element(errorZ.begin(), errorZ.end());

        std::cout << "Max Error in Gradient (X): " << maxErrorX << std::endl;
        std::cout << "Max Error in Gradient (Y): " << maxErrorY << std::endl;
        std::cout << "Max Error in Gradient (Z): " << maxErrorZ << std::endl;

        // here I compute an error density out of curiosity
        //  Calculate the error density for each component (X, Y, Z)
        errorDensityX = 0.0;
        errorDensityY = 0.0;
        errorDensityZ = 0.0;

        for (int i = 0; i < nxyz; i++)
        {
            errorDensityX += std::abs(analytic_ddx[i] - d_dx[i]);
            errorDensityY += std::abs(analytic_ddy[i] - d_dy[i]);
            errorDensityZ += std::abs(analytic_ddz[i] - d_dz[i]);
        }

        // Divide the sum of errors by the total number of grid points
        errorDensityX /= static_cast<double>(nxyz);
        errorDensityY /= static_cast<double>(nxyz);
        errorDensityZ /= static_cast<double>(nxyz);

        std::cout << "Error Density (X): " << errorDensityX << std::endl;
        std::cout << "Error Density (Y): " << errorDensityY << std::endl;
        std::cout << "Error Density (Z): " << errorDensityZ << std::endl;

        // clean up fftw plans
        fftw_destroy_plan(plan_f);
        fftw_destroy_plan(plan_b);

        // free the events
        cudaEventDestroy(g_start);
        cudaEventDestroy(g_stop);

        // clean up the gpu memory
        cudaFree(g_xyz);
        cudaFree(g_kxyz);
        cudaFree(g_fft3);
        cudaFree(g_wfft);
        cudaFree(g_ddx);
        cudaFree(g_ddy);
        cudaFree(g_ddz);
    }
    avg_gpu_time_ms /= static_cast<double>(ntrials);
    avg_cpu_time_s /= static_cast<double>(ntrials);
    results << avg_gpu_time_ms/1000. << ", "
            << 1000. * complexity / avg_gpu_time_ms << ", "
            << avg_cpu_time_s << ", "
            << complexity / avg_cpu_time_s << std::endl;
}

int main(int argc, char *argv[])
{
    std::ofstream results("./artifacts/p3-results.csv");
    results << "n, complexity, GPU Time (s), GPU FLOPs, CPU Time (s),"
        " CPU FLOPs" << std::endl;
    int ntrials = 5;
    for(int n=16; n<=256; n*=2)
    {
        measure_gradient_performance(n, ntrials, results);
    }
    results.close();
    return 0;
}
