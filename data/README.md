File descriptions:

All files are in HDF5 format.

*emd_dist_matrix_all_promoters_oct_20_2019*
    Earth Mover's Distance matrix of data shown in Fig 2d, i.e. all _tagged_ data from the "cell_types" Mesmerize project except for data from the eef1a promoter. Used for creating Fig 2b & 2c
    
    Opening the file:
        The dataset key is "data". If you're using Python you can open it using h5py: https://pypi.org/project/h5py/
        ```python
        import h5py
        f = h5py.File('./emd_dist_matrix_all_promoters_oct_20_2019.h5', 'r')
        dist_m = f['data']
        ```

