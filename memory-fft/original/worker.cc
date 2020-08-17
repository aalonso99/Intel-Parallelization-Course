#include <mkl.h>


//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {
  for(size_t i = 0; i < num_fft; i++) {
    DftiComputeForward (*fftHandle, &data[i*fft_size]);
  }
}