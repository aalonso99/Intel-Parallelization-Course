#include <mkl.h>
#include <hbwmalloc.h>
//#include <cstring>
#include <omp.h>
#include <iostream>

#define CHUNK_SIZE 8

//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {
    //Create buffer in high-bandwidth memory
    MKL_Complex8 *scratch_buffer;
    hbw_posix_memalign((void**) &scratch_buffer, 4096, sizeof(MKL_Complex8)*fft_size*CHUNK_SIZE);
    for(size_t i = 0; i < num_fft/CHUNK_SIZE; i++) {

        //Copy data for FFT
        #pragma omp parallel for
        for(size_t j = 0; j<fft_size*CHUNK_SIZE; j++){
            //memcpy(scratch_buffer, &data[i*fft_size], sizeof(MKL_Complex8));
            scratch_buffer[j] = data[i*fft_size*CHUNK_SIZE+j];
        }

        //Compute FFT
        for(size_t j = 0; j<CHUNK_SIZE; j++){
            DftiComputeForward (*fftHandle, &scratch_buffer[j*fft_size]);
        }

        //Copy solution back
        #pragma omp parallel for
        for(size_t j = 0; j<fft_size*CHUNK_SIZE; j++){
            //memcpy( &data[i*fft_size], scratch_buffer, sizeof(MKL_Complex8) );
            data[i*fft_size*CHUNK_SIZE+j] = scratch_buffer[j];
        }
        
    }

    hbw_free(scratch_buffer);
}