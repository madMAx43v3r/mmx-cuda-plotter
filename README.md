# mmx-cuda-plotter

CUDA plotter to create plots for MMX.

## Usage
Example Full RAM: `./mmx_cuda_plot_k30 -C 5 -n -1 -t /mnt/tmp_ssd/ -d <dst> -f <farmer_key>`

Example Partial RAM: `./mmx_cuda_plot_k30 -C 5 -n -1 -2 /mnt/tmp_ssd/ -d <dst> -f <farmer_key>`

Example Disk Mode: `./mmx_cuda_plot_k30 -C 5 -n -1 -3 /mnt/tmp_ssd/ -d <dst> -f <farmer_key>`

Usage is the same as for Gigahorse (https://github.com/madMAx43v3r/chia-gigahorse/tree/master/cuda-plotter), just without `-p` pool key.

If you have a fast CPU ([passmark benchmark](https://www.cpubenchmark.net) > 5000 points) you can use `-C 10` for HDD plots.

To create SSD plots (for farming on SSDs) add `--ssd` to the command and use `-C 0`.
SSD plots are 250% more efficient but cannot be farmed on HDDs. They have higher CPU load to farm, hence it's recommended to plot uncompressed.
