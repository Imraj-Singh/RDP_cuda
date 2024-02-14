from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='rdp',
    ext_modules=[
        cpp_extension.CUDAExtension('rdp', [
            'RelativeDifferencePrior.cu',
        ]),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })