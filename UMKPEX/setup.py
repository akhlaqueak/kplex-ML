from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


ext_modules = [
    Pybind11Extension(
        "UMKP._core",
        ["python/UMKP/_core.cpp"],
        include_dirs=["."],
        cxx_std=11,
        extra_compile_args=["-O3", "-w", "-DNO_PROGRESS_BAR", "-DNDEBUG"],
    )
]


setup(
    name="UMKP",
    version="0.1.0",
    description="Python package for the UMKPEX maximum k-plex solver",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    entry_points={
        "console_scripts": [
            "UMKP=UMKP.cli:main",
        ]
    },
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
