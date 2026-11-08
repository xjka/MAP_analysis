# ACJ:
# This script is for a record of building on franklin so I don't have to remember when I want to do it again.
# Had to install openmpi and fftw (because not in conda-packages.txt) as well as
#conda install openmpi fftw 

#NOTE: ENABLE_NUMA_FOR should depend on HAVE_LIBNUMA in the configure system. Don't have numa so for now am disabling

#LDFLAGS includes -I"" flags because they won't be put in right places by configure logic by only specifiying CPPFLAGS

make distclean
rm autom4te.cache -r
#Had to do this because was syntax error in configure straight from github (should probably flag that)
./autogen-config.sh

#configure command:
#NOTE: spherical overdensity and gpu directs have to be set to the same value. This should probably be enforced in the code to prevent error. 
./configure --disable-debug --enable-numa-for --enable-gpu-directs --disable-spherical-overdensity --enable-fof-density  --enable-tracers --prefix=$HOME/abacus --with-cuda=/usr/local/cuda-12.4

#amazingly, had to fix the actual table-logger python libary as it uses the deprecated np.float, np.int, etc. (which became deprecated in numpy 1.20 and an error in numpy 1.24) in:
#/home/ajohnson/miniconda3/lib/python3.12/site-packages/table_logger/table_logger.py

#had to install conda to have python dependencies (installed pybind11 with: "pip install pybind11") 

#had to downgrade asdf package so that it had asdf.compression module: pip3 install asdf==3.0.0

#had to install numpy with conda to get all needed headers for Corrfunc: "conda install numpy"

make
