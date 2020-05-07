
```console
# Create conda environment "info_projet9"
$ ./update_env.sh

# Activate this environment
$ conda activate "info_projet9"

# Install external libraries in the conda environment
$ ./extern/install_libs_in_env.sh

# Install info9 package in the conda environment
$ python -m pip install "../pkg_info9"

# Run Jupyter Notebooks in the conda environment
$ ./notebook.sh
```
