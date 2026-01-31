import torch
import numpy as np
import threading
import gc
import math
import copy
import sys
# Import the custom extension
try:
    import mmNorm
    # The extension module might be under mmNorm or directly importable depending on setup.py
    # From setup.py: name='mmNorm', packages=find_packages(), ext_modules=[name='imaging_gpu'...]
    # So it should be `import imaging_gpu` or `from mmNorm import imaging_gpu`? 
    # Usually valid extension names at top level.
    # We will assume 'imaging_gpu' is importable.
except ImportError:
    pass # Will check later

# Helper to check if extension is loaded
def get_imaging_extension():
    try:
        import imaging_gpu
        return imaging_gpu
    except ImportError:
        # Fallback if installed as package
        try:
            from mmNorm import imaging_gpu
            return imaging_gpu
        except ImportError:
            raise ImportError("Could not import imaging_gpu extension. Make sure to install the package.")

class GPUThread(threading.Thread):
    def __init__(self, device_id, 
                 x_locs, y_locs, z_locs, antenna_locs_flat, 
                 meas_real, meas_imag, rx_offset, 
                 slope, wavelength, fft_spacing, 
                 num_x, num_y, num_z, num_ant, num_rx_ant, 
                 samples_per_meas, start_ind, is_ti_radar, use_interpolated_processing):
        threading.Thread.__init__(self)
        self.device_id = device_id
        self.x_locs = x_locs
        self.y_locs = y_locs
        self.z_locs = z_locs
        self.antenna_locs_flat = antenna_locs_flat
        self.meas_real = meas_real
        self.meas_imag = meas_imag
        self.rx_offset = rx_offset
        self.slope = slope
        self.wavelength = wavelength
        self.fft_spacing = fft_spacing
        self.num_x = num_x
        self.num_y = num_y
        self.num_z = num_z
        self.num_ant = num_ant
        self.num_rx_ant = num_rx_ant
        self.samples_per_meas = samples_per_meas
        self.start_ind = start_ind
        self.is_ti_radar = is_ti_radar
        self.use_interpolated_processing = use_interpolated_processing
        
        self.result_r = None
        self.result_i = None

    def run(self):
        with torch.cuda.device(self.device_id):
            ext = get_imaging_extension()
            
            # Transfer data to GPU
            # Note: We move copies to the specific device
            d_x_locs = torch.from_numpy(self.x_locs).float().cuda()
            d_y_locs = torch.from_numpy(self.y_locs).float().cuda()
            d_z_locs = torch.from_numpy(self.z_locs).float().cuda()
            d_antenna_locs = torch.from_numpy(self.antenna_locs_flat).float().cuda()
            d_meas_real = torch.from_numpy(self.meas_real).float().cuda()
            d_meas_imag = torch.from_numpy(self.meas_imag).float().cuda()
            d_rx_offset = torch.from_numpy(self.rx_offset).float().cuda()
            
            # Alloc output
            d_p_xyz_r = torch.zeros(self.num_x * self.num_y * self.num_z, dtype=torch.float32, device='cuda')
            d_p_xyz_i = torch.zeros(self.num_x * self.num_y * self.num_z, dtype=torch.float32, device='cuda')

            ext.cuda_image(
                d_p_xyz_r, d_p_xyz_i,
                d_x_locs, d_y_locs, d_z_locs,
                d_antenna_locs, d_meas_real, d_meas_imag,
                d_rx_offset,
                self.slope, self.wavelength, self.fft_spacing,
                self.num_x, self.num_y, self.num_z, 
                self.num_ant, self.num_rx_ant, self.samples_per_meas,
                self.start_ind, self.is_ti_radar, self.use_interpolated_processing
            )
            
            # Copy back to CPU
            self.result_r = d_p_xyz_r.cpu().numpy()
            self.result_i = d_p_xyz_i.cpu().numpy()
            
            del d_x_locs, d_y_locs, d_z_locs, d_antenna_locs, d_meas_real, d_meas_imag, d_rx_offset
            del d_p_xyz_r, d_p_xyz_i
            torch.cuda.empty_cache()

    def get_res(self):
        return self.result_r, self.result_i

def _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx):
    if use_4_rx:
        radar_data = np.transpose(radar_data, (0,2,1)) 
        antenna_locs = np.repeat(antenna_locs, radar_data.shape[1], axis=0)
    rx_offset = np.array(rx_offset)
    measurements = radar_data.reshape((-1))
    return rx_offset, measurements, antenna_locs

def _run_cuda_image(x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, use_interpolated_processing=False):
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} CUDA Capable device(s)')

    norm_factor = antenna_locs.shape[0]
    samples_per_meas = radar_data.shape[1]
    rx_offset_flat, measurements, antenna_locs = _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx)
    
    # Ensure float32
    meas_real = np.ascontiguousarray(measurements.real).astype(np.float32)
    meas_imag = np.ascontiguousarray(measurements.imag).astype(np.float32)
    rx_offset_flat = rx_offset_flat.flatten().astype(np.float32)
    antenna_locs_flat = np.array(antenna_locs).flatten().astype(np.float32)
    x_locs = np.array(x_locs).astype(np.float32)
    y_locs = np.array(y_locs).astype(np.float32)
    z_locs = np.array(z_locs).astype(np.float32)

    num_x = len(x_locs)
    num_y = len(y_locs)
    num_z = len(z_locs)
    num_ant = len(antenna_locs)
    
    # Bug fix logic from original: rx_offset shape handling
    # rx_offset comes in as (#RX, 3) from caller usually, _prep_inputs returns rx_offset but flattened?
    # Original _prep_inputs returned 'rx_offset' as np array. Flatten happened later in _run_cuda_image
    # Here we flattened it above.
    num_rx_ant = len(rx_offset) # Use original passed in rx_offset to get count

    fft_spacing = np.float32(3e8/(2*bandwidth)*num_samples/(radar_data.shape[1]))

    threads_per_block = 512
    # Original grid_dim calculation
    grid_dim = int(num_x*num_y*num_z / threads_per_block / num_gpus) + 1
    
    print(f'Starting GPU computation')
    gpu_thread_list = []
    
    for i in range(num_gpus):
        start_ind = int(grid_dim) * int(threads_per_block) * i
        t = GPUThread(i, x_locs, y_locs, z_locs, antenna_locs_flat, 
                      meas_real, meas_imag, rx_offset_flat, 
                      slope, wavelength, fft_spacing, 
                      num_x, num_y, num_z, num_ant, num_rx_ant, 
                      samples_per_meas, start_ind, 
                      1 if is_ti_radar else 0, 
                      1 if use_interpolated_processing else 0)
        t.start()
        gpu_thread_list.append(t)

    p_xyz_r_list = []
    p_xyz_i_list = []
    for thread in gpu_thread_list:
        thread.join()
        res_r, res_i = thread.get_res()
        p_xyz_r_list.append(res_r)
        p_xyz_i_list.append(res_i)

    # Sum outputs from all GPUs
    # Note: original code summed them up. Since we process same voxels but potentially different ranges?
    # Wait, the kernel checks: if (ind >= NUM_X*NUM_Y*NUM_Z ... return).
    # And ind = blockIdx... + start_ind.
    # So each GPU processes a *subset* of voxels?
    # If they process a subset, the outputs will be zeroes elsewhere?
    # The kernel initializes d_p_xyz output to 0?
    # My wrapper uses torch.zeros.
    # So summing them up is correct IF they cover disjoint sets of voxels.
    
    p_xyz_r = np.sum(p_xyz_r_list, axis=0)
    p_xyz_i = np.sum(p_xyz_i_list, axis=0)

    image_shape = (num_x, num_y, num_z)
    p_xyz_r = p_xyz_r.reshape((image_shape))
    p_xyz_i = p_xyz_i.reshape((image_shape))
    p_xyz = p_xyz_r + 1j*p_xyz_i
    p_xyz /= norm_factor 

    return p_xyz, norm_factor, antenna_locs

def image_cuda(x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offsets, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, use_interpolated_processing=False, return_phase_centers=True):
    sum_image = None
    if not use_interpolated_processing:
        sum_image, norm_factor, locs_rep = _run_cuda_image(x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offsets, slope, wavelength, bandwidth, num_samples, use_4_rx, is_ti_radar, use_interpolated_processing)
        sum_image /= norm_factor
        
        # Phase center logic
        rx_locs = copy.deepcopy(locs_rep)
        rx_offsets_arr = np.array(rx_offsets)
        rx_offset_rep = np.tile(rx_offsets_arr, reps=(len(locs_rep)//len(rx_offsets_arr),1))
        rx_locs += rx_offset_rep
        center_locs = (locs_rep + rx_locs)/2
    else:
        max_num_ant = 4096*32 
        num_rx = len(rx_offsets)
        # Fix possible division by zero or float
        num_ant_groups = math.ceil(antenna_locs.shape[0]*num_rx/max_num_ant)

        for i in range(num_ant_groups):
            start_idx = i*max_num_ant//num_rx
            end_idx = (i+1)*max_num_ant//num_rx
            loc_subset = antenna_locs[start_idx:end_idx]
            meas_subset = radar_data[start_idx:end_idx]
            
            image_subset, norm_factor, locs_rep = _run_cuda_image(x_locs, y_locs, z_locs, loc_subset, meas_subset, rx_offsets, slope, wavelength, bandwidth, num_samples, use_4_rx, is_ti_radar, use_interpolated_processing)
            
            if return_phase_centers:
                rx_locs = copy.deepcopy(locs_rep)
                rx_offsets_arr = np.array(rx_offsets)
                rx_offset_rep = np.tile(rx_offsets_arr, reps=(len(locs_rep)//len(rx_offsets_arr),1))
                rx_locs += rx_offset_rep
                center_locs_subset = (locs_rep + rx_locs)/2

            if i == 0:
                sum_image = image_subset
                if return_phase_centers:
                    center_locs = center_locs_subset
            else:
                sum_image += image_subset
                if return_phase_centers:
                    center_locs = np.concatenate((center_locs, center_locs_subset), axis=0)

    if return_phase_centers:
        return sum_image, center_locs
    return sum_image
