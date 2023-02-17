from setuptools import setup, find_packages
from torch.utils import cpp_extension

spherical_harmonics_extension = cpp_extension.CppExtension("spherical_harmonics_extension", 
      ["sphericart_torch/extension/spherical_harmonics.cc"], 
      extra_compile_args=["-std=c++17", "-O3", "-fopenmp", "-Wall"])

ext_modules = [spherical_harmonics_extension]

setup(name="sphericart_torch",
      packages = find_packages(),
      ext_modules = ext_modules,
      cmdclass={"build_ext": cpp_extension.BuildExtension})
