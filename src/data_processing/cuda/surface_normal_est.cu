#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define SPEED_LIGHT 2.99792458e8
const double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

#define mult_r(a,b,c,d)(a*c-b*d)
#define mult_i(a,b,c,d)(a*d+b*c)
  
extern "C"
__global__ void cuda_sn_est_kernel(float* device_normals_x, float* device_normals_y, float* device_normals_z, 
                            const int* valid_idx_x, const int* valid_idx_y, const int* valid_idx_z,
                           const float* device_p_xyz_r, 
                           const float* device_p_xyz_i, 
                           const float* device_x_locs,
                           const float* device_y_locs,
                           const float* device_z_locs,
                           const float* device_antenna_locs, 
                           const float* device_measurements_r,
                           const float* device_measurements_i, 
                           const float* rx_offsets, float slope, float wavelength, float fft_spacing, 
                           int NUM_X, int NUM_Y, int NUM_Z, int NUM_ANTENNAS, int NUM_RX_ANTENNAS, int NUM_VALID_IDX, int SAMPLES_PER_MEAS, int start_ind, int is_ti_radar ) {
    
    // Find the current index from the GPU block/thread index
    int ind = (blockIdx.x * blockDim.x + threadIdx.x) + start_ind;
    if (ind >= NUM_VALID_IDX || ind < 0) { return; }

    // Convert the index to the location of a valid voxel
    int x = valid_idx_x[ind];
    int y = valid_idx_y[ind];
    int z = valid_idx_z[ind];
    float x_loc = device_x_locs[x];
    float y_loc = device_y_locs[y];
    float z_loc = device_z_locs[z];

    // Read the SAR value of this voxel
    float image_r = device_p_xyz_r[x*NUM_Y*NUM_Z+y*NUM_Z+z];
    float image_i = device_p_xyz_i[x*NUM_Y*NUM_Z+y*NUM_Z+z];
    float image_mag = sqrtf(image_r * image_r + image_i * image_i);

    // Compute surface normal for this voxel. Need to compute weighted sum across all antennas
    float sn_x = 0;
    float sn_y = 0;
    float sn_z = 0;
    for (uint i = 0; i < NUM_ANTENNAS; i++) {
        // Find location of TX/RX antenna
        float x_antenna_loc = device_antenna_locs[i*3+0];
        float y_antenna_loc = device_antenna_locs[i*3+1];
        float z_antenna_loc = device_antenna_locs[i*3+2];
        float antenna_x_diff = x_loc - x_antenna_loc;
        float antenna_y_diff = y_loc - y_antenna_loc;
        float antenna_z_diff = z_loc - z_antenna_loc;
        int rx_num = i%NUM_RX_ANTENNAS;
        float rx_offset_x = rx_offsets[rx_num*3+0];
        float rx_offset_y = rx_offsets[rx_num*3+1];
        float rx_offset_z = rx_offsets[rx_num*3+2];

        // ---------- Compute SAR image component using this single antenna (Eq. 4 in paper) -------------------
        // Find distance from TX -> voxel -> RX
        float forward_dist = sqrtf(antenna_x_diff * antenna_x_diff + 
                                        antenna_y_diff * antenna_y_diff + 
                                        antenna_z_diff * antenna_z_diff);
        float back_dist = sqrtf((antenna_x_diff - rx_offset_x)* (antenna_x_diff - rx_offset_x) + 
                                    (antenna_y_diff - rx_offset_y) * (antenna_y_diff - rx_offset_y) + 
                                    (antenna_z_diff - rx_offset_z) * (antenna_z_diff - rx_offset_z));
        float distance = forward_dist + back_dist;
        if (is_ti_radar != 0){ 
            distance += 0.15; // The TI radars have an offset of 15cm that needs to be accounted for
        }

        // Check if distance is valid
        if (distance < 0 || distance > fft_spacing*SAMPLES_PER_MEAS) {
            continue;
        }

        // Find which bin in FFT corresponds to this distance
        int dist_bin = floorf(distance / fft_spacing/2);

        // Select the appropriate measurement, and coorelate with the AoA phase
        float real_meas = device_measurements_r[i*SAMPLES_PER_MEAS+dist_bin];
        float imag_meas = device_measurements_i[i*SAMPLES_PER_MEAS+dist_bin];
        float real_phase = cosf(-2 * pi * distance / wavelength);
        float imag_phase = sinf(-2 * pi * distance / wavelength);
        float sum_r = mult_r(real_meas, imag_meas, real_phase, imag_phase);
        float sum_i = mult_i(real_meas, imag_meas, real_phase, imag_phase);

        // ---------- Compute weighted candidate normal vector and add to current sum (Eq. 3/6 in paper) -------------------
        // Virtual antenna between TX/RX
        float virtual_antenna_x =  x_antenna_loc + rx_offset_x / 2; 
        float virtual_antenna_y =  y_antenna_loc + rx_offset_y / 2; 
        float virtual_antenna_z =  z_antenna_loc + rx_offset_z / 2; 

        // Vector from antenna to voxel location
        float vec_x = virtual_antenna_x - x_loc;
        float vec_y = virtual_antenna_y - y_loc;
        float vec_z = virtual_antenna_z - z_loc;

        // Compute vote (Eq. 6)
        float weight = (sum_r * image_r) + (sum_i * image_i);
        weight /= image_mag;

        // Add weighted candidate vector to current weighted sum
        sn_x += (vec_x*weight);
        sn_y += (vec_y*weight);
        sn_z += (vec_z*weight);
    }
    // Save normal estimate
    device_normals_x[ind] = sn_x;
    device_normals_y[ind] = sn_y;
    device_normals_z[ind] = sn_z;
}

void cuda_sn_est_launcher(
    torch::Tensor normals_x,
    torch::Tensor normals_y,
    torch::Tensor normals_z,
    torch::Tensor valid_idx_x,
    torch::Tensor valid_idx_y,
    torch::Tensor valid_idx_z,
    torch::Tensor p_xyz_r,
    torch::Tensor p_xyz_i,
    torch::Tensor x_locs,
    torch::Tensor y_locs,
    torch::Tensor z_locs,
    torch::Tensor antenna_locs_flat,
    torch::Tensor meas_real,
    torch::Tensor meas_imag,
    torch::Tensor rx_offset,
    float slope,
    float wavelength,
    float fft_spacing,
    int num_x, int num_y, int num_z,
    int num_ant, int num_rx_ant,
    int num_valid_idx,
    int samples_per_meas,
    int start_ind,
    int is_ti_radar
) {
    CHECK_INPUT(normals_x);
    CHECK_INPUT(normals_y);
    CHECK_INPUT(normals_z);
    CHECK_INPUT(valid_idx_x);
    CHECK_INPUT(valid_idx_y);
    CHECK_INPUT(valid_idx_z);
    CHECK_INPUT(p_xyz_r);
    CHECK_INPUT(p_xyz_i);
    CHECK_INPUT(x_locs);
    CHECK_INPUT(y_locs);
    CHECK_INPUT(z_locs);
    CHECK_INPUT(antenna_locs_flat);
    CHECK_INPUT(meas_real);
    CHECK_INPUT(meas_imag);
    CHECK_INPUT(rx_offset);

    int threads_per_block = 512;
    int blocks = (num_valid_idx + threads_per_block - 1) / threads_per_block;

    cuda_sn_est_kernel<<<blocks, threads_per_block>>>(
        normals_x.data_ptr<float>(),
        normals_y.data_ptr<float>(),
        normals_z.data_ptr<float>(),
        valid_idx_x.data_ptr<int>(),
        valid_idx_y.data_ptr<int>(),
        valid_idx_z.data_ptr<int>(),
        p_xyz_r.data_ptr<float>(),
        p_xyz_i.data_ptr<float>(),
        x_locs.data_ptr<float>(),
        y_locs.data_ptr<float>(),
        z_locs.data_ptr<float>(),
        antenna_locs_flat.data_ptr<float>(),
        meas_real.data_ptr<float>(),
        meas_imag.data_ptr<float>(),
        rx_offset.data_ptr<float>(),
        slope,
        wavelength,
        fft_spacing,
        num_x, num_y, num_z, num_ant, num_rx_ant,
        num_valid_idx, samples_per_meas, start_ind, is_ti_radar
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_sn_est", &cuda_sn_est_launcher, "Surface Normal Est GPU Kernel");
}