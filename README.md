# GemsDiff

Official implementation of GemsDiff. However, we realised after publication that our proposed model is more efficient so we provided the newest version without lattice diffusion. This repo contains dataset and checkpoints to retrain and generate materials.

All scripts can run on CPU and CUDA GPU but cuda is unsed by default. Please follow documentation to install pytorch and torch-geometric on CPU only.

## Installation on a virtual environement

Create and activate the environement

```bash
python3 -m venv gemsdiff
source gemsdiff/bin/activate
```

Installing pytorch and torch geometric (see documentation: [pytorch](https://pytorch.org/get-started/locally/) and [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))

```bash
pip3 install torch
pip3 install torch_geometric
pip3 install torch_scatter
```

Install Crystallographic graph [Crystallographic graph](https://github.com/aklipf/mat-graph)

```bash
pip3 install git+https://gitlab.com/chem-test/crystallographic-graph.git 
```

Install other dependancies

```bash
pip3 install torch-ema pandas tqdm matplotlib h5py pymatgen ase tensorboard
```

## Sampling a specific composition

Sampling LiFeO2 from a checkpoint (OQMD)

```bash
python3 sampling.py LiFeO2 -c runs/without_cell_diffusion/training_2024_02_23_16_13_55_oqmd_604 -o LiFeO2.cif
```

## Sampling structures of a given system from checkpoint

Sampling structure from the Li-Fe-O system from a checkpoint (Materials Project)

```bash
python3 sampling_system.py Li-Fe-O -c runs/without_cell_diffusion/training_2024_02_23_01_40_14_mp_110 -o Li-Fe-O.cif
```

## How to cite

```bibtex
@article{klipfel_diff_aaai_2024,
    author={Astrid Klipfel and Ya{\"{e}}l Fr{\'{e}}gier and Adlane Sayede and Zied Bouraoui},
    title={Vector Field Oriented Diffusion Model for Crystal Material Generation},
    year={2024},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}
}
```
