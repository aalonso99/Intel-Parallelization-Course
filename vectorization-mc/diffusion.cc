#include <mkl.h>
#include <cstring>
#include "distribution.h"


//vectorize this function based on instruction on the lab page
int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {

    int n_escaped=0;
    float* x = new float [n_particles];
    memset( x, 0.0f, n_particles*sizeof(float) );
    float* rn = new float [n_particles];

    for (int j = 0; j < n_steps; j++) {

        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,
                    rnStream, n_particles, rn, -1.0, 1.0);

        for (int i = 0; i < n_particles; i++) {
            
            x[i] += dist_func(alpha, rn[i]); 

        }

    }

    for (int i=0; i<n_particles; i++){
        if (x[i] > x_threshold) n_escaped++;
    }

    delete [] x;
    delete [] rn;

    return n_escaped;
}