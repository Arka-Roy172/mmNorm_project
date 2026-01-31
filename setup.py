
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Define the extensions
ext_modules = [
    # C++ Extension for Imaging
    CppExtension(
        name='imaging',
        sources=['src/data_processing/cpp/imaging.cpp'],
        include_dirs=['src/data_processing/cpp', 'src/utils/include'],
        extra_compile_args=['-O3', '-std=c++14', '-fopenmp', '-fPIC']
    ),
    # CUDA Extension for Imaging GPU
    CUDAExtension(
        name='imaging_gpu',
        sources=['src/data_processing/cuda/imaging_gpu.cu'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': ['-O3', '--use_fast_math', '-arch=sm_86']
        }
    ),
     # CUDA Extension for Surface Normal Estimation
    CUDAExtension(
        name='surface_normal_est',
        sources=['src/data_processing/cuda/surface_normal_est.cu'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': ['-O3', '--use_fast_math', '-arch=sm_86']
        }
    ),
     # Existing Ray Tracing Extension
    CUDAExtension(
        name='ray_tracing',
        sources=[
            'src/data_processing/cpp/ray_tracing.cpp', # Defines the module entry point
            'src/data_processing/cuda/ray_tracing_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': ['-O3', '-arch=sm_86']
        }
    ),
]

setup(
    name='mmNorm',
    version='0.1.0',
    description='Non-Line-of-Sight 3D Object Reconstruction via mmWave Surface Normal Estimation',
    packages=find_packages(where='src'), # Use src as package root if possible
    package_dir={'': 'src'}, # Map root to src
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.5.0',
        'numpy>=1.24.0',
        'torchvision',
        'scipy', 
        'tqdm'
    ]
)
