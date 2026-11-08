#!/bin/bash

 Builds pfft-python==0.1.24 on MPICH-based systems (e.g. Aurora)
 # where the default source fails due to MPI_Comm ABI incompatibility.
 #
 # Problem: pfft/core.c assigns NULL to MPI_Comm fields, which is valid
 # under OpenMPI (pointer type) but fails under MPICH (int type).
 # Fix: patch core.pyx to use MPI_COMM_NULL instead, then let setuptools
 # regenerate core.c via Cython during the build.

 set -e

 VERSION="0.1.24"
 TARBALL="pfft-python-${VERSION}.tar.gz"
 SRCDIR="pfft-python-${VERSION}"

 echo "==> Downloading pfft-python==${VERSION} source..."
 pip download --no-deps --no-binary pfft-python "pfft-python==${VERSION}"

 echo "==> Extracting..."
 tar xzf "${TARBALL}"
 cd "${SRCDIR}"

 echo "==> Patching pfft/core.pyx for MPICH MPI_Comm ABI compatibility..."

 # self.ccart = NULL  ->  self.ccart = cMPI.MPI_COMM_NULL
 # (MPI_Comm is int under MPICH, not a pointer; NULL is invalid)
 sed -i 's/self\.ccart = NULL/self.ccart = cMPI.MPI_COMM_NULL/g' pfft/core.pyx

 # self.ccol[i] = NULL  ->  self.ccol[i] = cMPI.MPI_COMM_NULL
 # (individual communicator slots in the ccol array)
 sed -i 's/self\.ccol\[i\] = NULL/self.ccol[i] = cMPI.MPI_COMM_NULL/g' pfft/core.pyx

 # self.ccol = NULL is a pointer-to-MPI_Comm array — NULL is correct here,
 # but needs an explicit cast to suppress compiler warnings under MPICH
 sed -i 's/self\.ccol = NULL/self.ccol = <cMPI.MPI_Comm*>NULL/g' pfft/core.pyx

 echo "==> Verifying patches..."
 grep -n "ccart\|ccol" pfft/core.pyx | grep -i null

 echo "==> Building and installing (no build isolation, uses system MPI)..."
 uv pip install --no-build-isolation .

 echo "==> Done."
