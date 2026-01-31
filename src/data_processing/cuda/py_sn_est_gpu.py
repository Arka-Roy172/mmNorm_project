import torch
import numpy as np
import threading
import gc
import math
import copy

def get_sn_est_extension():
    try:
        import surface_normal_est
        return surface_normal_est
    except ImportError:
        try:
            from mmNorm import surface_normal_est
            return surface_normal_est
        except ImportError:
            raise ImportError("Could not import surface_normal_est extension.")

class GPUThread(threading.Thread):
    def __init__(self, device_id, 
                 sum_image_r, sum_image_i, 
                 valid_idx_x, valid_idx_y, valid_idx_z,
                 x_locs, y_locs, z_locs, antenna_locs_flat,
                 meas_real, meas_imag, rx_offset_flat,
                 slope, wavelength, fft_spacing,
                 num_x, num_y, num_z, num_ant, num_rx_ant,
                 num_valid_idx, samples_per_meas,
                 start_ind, is_ti_radar):
        threading.Thread.__init__(self)
        self.device_id = device_id
        self.sum_image_r = sum_image_r
        self.sum_image_i = sum_image_i
        self.valid_idx_x = valid_idx_x
        self.valid_idx_y = valid_idx_y
        self.valid_idx_z = valid_idx_z
        self.x_locs = x_locs
        self.y_locs = y_locs
        self.z_locs = z_locs
        self.antenna_locs_flat = antenna_locs_flat
        self.meas_real = meas_real
        self.meas_imag = meas_imag
        self.rx_offset_flat = rx_offset_flat
        self.slope = slope
        self.wavelength = wavelength
        self.fft_spacing = fft_spacing
        self.num_x = num_x
        self.num_y = num_y
        self.num_z = num_z
        self.num_ant = num_ant
        self.num_rx_ant = num_rx_ant
        self.num_valid_idx = num_valid_idx
        self.samples_per_meas = samples_per_meas
        self.start_ind = start_ind
        self.is_ti_radar = is_ti_radar
        
        self.nx = None
        self.ny = None
        self.nz = None

    def run(self):
        with torch.cuda.device(self.device_id):
            ext = get_sn_est_extension()
            
            # Move to GPU
            d_nx = torch.zeros(self.num_valid_idx, dtype=torch.float32, device='cuda')
            d_ny = torch.zeros(self.num_valid_idx, dtype=torch.float32, device='cuda')
            d_nz = torch.zeros(self.num_valid_idx, dtype=torch.float32, device='cuda')
            
            d_idx_x = torch.from_numpy(self.valid_idx_x).int().cuda()
            d_idx_y = torch.from_numpy(self.valid_idx_y).int().cuda()
            d_idx_z = torch.from_numpy(self.valid_idx_z).int().cuda()
            
            d_p_r = torch.from_numpy(self.sum_image_r).float().cuda()
            d_p_i = torch.from_numpy(self.sum_image_i).float().cuda()
            
            d_x = torch.from_numpy(self.x_locs).float().cuda()
            d_y = torch.from_numpy(self.y_locs).float().cuda()
            d_z = torch.from_numpy(self.z_locs).float().cuda()
            d_ant = torch.from_numpy(self.antenna_locs_flat).float().cuda()
            d_mr = torch.from_numpy(self.meas_real).float().cuda()
            d_mi = torch.from_numpy(self.meas_imag).float().cuda()
            d_rx = torch.from_numpy(self.rx_offset_flat).float().cuda()
            
            ext.cuda_sn_est(
                d_nx, d_ny, d_nz,
                d_idx_x, d_idx_y, d_idx_z,
                d_p_r, d_p_i,
                d_x, d_y, d_z, d_ant,
                d_mr, d_mi, d_rx,
                self.slope, self.wavelength, self.fft_spacing,
                self.num_x, self.num_y, self.num_z,
                self.num_ant, self.num_rx_ant,
                self.num_valid_idx, self.samples_per_meas,
                self.start_ind, self.is_ti_radar
            )
            
            self.nx = d_nx.cpu().numpy()
            self.ny = d_ny.cpu().numpy()
            self.nz = d_nz.cpu().numpy()
            
            del d_nx, d_ny, d_nz, d_p_r, d_p_i, d_x, d_y, d_z, d_ant
            torch.cuda.empty_cache()
            
    def get_res(self):
        return self.nx, self.ny, self.nz

def _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx):
    if not use_4_rx: 
        rx_offset = np.array([rx_offset]) 
        radar_data = np.array(radar_data).flatten()
    else:
        # Reorder radar data to be (#Meas, #RXAntenna, #Sample/Meas)
        radar_data = np.transpose(radar_data, (0,2,1)) 
        rx_offset = np.array(rx_offset).flatten() 
        antenna_locs = np.repeat(antenna_locs, radar_data.shape[1], axis=0)

    return rx_offset, radar_data, antenna_locs

def _sn_est_cuda(sum_image, x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, initial_filter_percent=0.05):
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} CUDA Capable device(s)')
    
    samples_per_meas = radar_data.shape[1]
    rx_offset_flat, measurements, antenna_locs = _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx)
    
    meas_real = np.ascontiguousarray(measurements.real).astype(np.float32)
    meas_imag = np.ascontiguousarray(measurements.imag).astype(np.float32)
    antenna_locs_flat = np.array(antenna_locs).flatten().astype(np.float32)
    sum_image_r = np.real(sum_image).flatten().astype(np.float32)
    sum_image_i = np.imag(sum_image).flatten().astype(np.float32)
    rx_offset_flat = rx_offset_flat.astype(np.float32)

    # Filter valid voxels
    filter_idx = np.abs(sum_image) > (np.max(np.abs(sum_image))*initial_filter_percent)
    mesh1, mesh2, mesh3 = np.meshgrid(np.arange(len(y_locs)), np.arange(len(x_locs)), np.arange(len(z_locs)))
    coords_full = np.concatenate((mesh2[...,np.newaxis], mesh1[...,np.newaxis],mesh3[...,np.newaxis]), axis=-1)
    coords_full = coords_full[filter_idx]
    valid_idx_x = coords_full[:,0].astype(np.int32)
    valid_idx_y = coords_full[:,1].astype(np.int32)
    valid_idx_z = coords_full[:,2].astype(np.int32)
    num_valid_idx = valid_idx_y.shape[0]

    num_x = len(x_locs)
    num_y = len(y_locs)
    num_z = len(z_locs)
    num_ant = len(antenna_locs)
    num_rx_ant = len(rx_offset) if use_4_rx else 1
    
    fft_spacing = np.float32(3e8/(2*bandwidth)*num_samples/(radar_data.shape[1]))
    
    threads_per_block = 512
    # Grid dim covers the VALID indices
    grid_dim = int(num_valid_idx/ threads_per_block / num_gpus) + 1
    
    print(f'Starting GPU computation')
    gpu_thread_list = []
    
    for i in range(num_gpus):
        start_ind = int(grid_dim) * int(threads_per_block) * i
        t = GPUThread(i, sum_image_r, sum_image_i, 
                      valid_idx_x, valid_idx_y, valid_idx_z,
                      x_locs, y_locs, z_locs, 
                      antenna_locs_flat, meas_real, meas_imag, rx_offset_flat,
                      slope, wavelength, fft_spacing,
                      num_x, num_y, num_z, num_ant, num_rx_ant,
                      num_valid_idx, samples_per_meas,
                      start_ind, 1 if is_ti_radar else 0)
        t.start()
        gpu_thread_list.append(t)
        
    nx_list = []
    ny_list = []
    nz_list = []
    
    for thread in gpu_thread_list:
        thread.join()
        rx, ry, rz = thread.get_res()
        nx_list.append(rx)
        ny_list.append(ry)
        nz_list.append(rz)

    normals_x = np.sum(nx_list, axis=0) 
    normals_y = np.sum(ny_list, axis=0) 
    normals_z = np.sum(nz_list, axis=0) 

    normals = np.concatenate((normals_x[:,np.newaxis], normals_y[:,np.newaxis], normals_z[:,np.newaxis]), axis=-1)
    
    all_normals = np.zeros((num_x, num_y, num_z, 3))
    all_normals[:] = np.nan
    all_normals[filter_idx] = normals

    return all_normals

def normal_estimates_cuda(sum_image, x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, initial_filter_percent=0.05, use_interpolated_processing=True):
    if not use_interpolated_processing:
       raise Exception("Surface normal estimation is not currently implemented without interpolated processing!")
    else:
        max_num_ant = 4096*32 
        num_rx = len(rx_offset)
        num_ant_groups = math.ceil(antenna_locs.shape[0]*num_rx/max_num_ant)

        for i in range(num_ant_groups):
            start_idx = i*max_num_ant//num_rx
            end_idx = (i+1)*max_num_ant//num_rx
            loc_subset = antenna_locs[start_idx:end_idx]
            meas_subset = radar_data[start_idx:end_idx]
            
            normals_subset = _sn_est_cuda(sum_image, x_locs, y_locs, z_locs, loc_subset, meas_subset, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx, is_ti_radar, initial_filter_percent)
            
            if i == 0:
                normals = normals_subset
            else:
                # Summing normals? The original code did:
                # normals += normals_subset
                # Yes, they just sum the contributions.
                # However, normals contain NaN for invalid voxels. 
                # np.nansum might be safer but original used +=. 
                # Since NaN+anything = NaN, this implies the mask must be identical and values accum.
                # The mask depends on sum_image which is constant inside this loop?
                # Yes, sum_image is passed in.
                normals += normals_subset
        
    return normals
