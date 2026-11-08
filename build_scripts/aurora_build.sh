#. env/aurora-0D.sh
#meson setup build -Dgpu_api=sycl -Dtracers=false -Dmpi=disabled
#meson compile -C build
cd ~/abacus
. env/aurora.sh
#uv sync --reinstall-package abacus
uv pip install ~/abacusutils
uv pip install jupyterlab ipyparallel scikit-learn scikit-optimize hdfstream swiftsimio healpy nbodykit 
#Corrfunc/pycorr
uv pip install ~/pfft_python-0.1.24
uv pip install ~/Corrfunc
uv pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[mpi,jackknife,corrfunc]

#illustris_python
uv pip install ~/illustris_python

