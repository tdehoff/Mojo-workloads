## Quick start

* If you don't have `pixi`, you can install it with:
```
curl -fsSL https://pixi.sh/install.sh | sh
```
* Navigate to the workload directory and activate the virtual environment:
```
cd <workload_directory>/Mojo
pixi shell
```
* Run with:
```
mojo <workload>.mojo
```

## Citation

If you find this repo useful, please cite our [SC25-WACCPD](https://waccpd.org/) paper:

```
@INPROCEEDINGS{waccpd2025mojo,
  author    = {William F. Godoy and Tatiana Melnichenko and Pedro Valero-Lara and Wael Elwasif and Philip Fackler and Rafael Ferreira Da Silva and Keita Teranishi and Jeffrey S. Vetter},
  title     = {Mojo: {MLIR}-Based Performance-Portable {HPC} Science Kernels on {GPUs} for the {Python} Ecosystem},
  booktitle = {Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC Workshops '25)},
  year      = {2025},
  pages     = {15},
  publisher = {ACM},
  address   = {New York, NY, USA},
  location  = {St. Louis, MO, USA},
  month     = nov,
  doi       = {10.1145/3731599.3767573},
  url       = {https://doi.org/10.1145/3731599.3767573}
}
```