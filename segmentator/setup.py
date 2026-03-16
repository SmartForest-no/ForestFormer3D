from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='segmentator_ext',
    ext_modules=[
        CppExtension(
            name='libsegmentator',
            sources=['csrc/segmentator.cpp'],
            extra_compile_args=['-std=c++14']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)