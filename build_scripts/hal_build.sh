#!/bin/bash
# ACJ: This is a record of how I built on hal (with new Ubuntu 22.04 OS) so I don't
# have to remember how I did it. 

#rm -r build

#set -e 

source env/hal.sh

meson setup build -Dgpu_api=sycl -Dtracers=true -Dprefix=$HOME/abacus_sycl -Dmpi=disabled -Dzeldovich-PLT:DISK=true 

meson compile -C build  

#Additional set-up steps I did:
#uv python install 3.13
#uv python pin 3.13           # records the interpreter for this project
#uv sync                      # or uv pip install -e .
#these were to address the fact that at this time numba was not compatible with python 3.14, 
#so had to install compatible version here. 
