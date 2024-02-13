from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='add_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension('add_cuda_ext', [
            'add_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })