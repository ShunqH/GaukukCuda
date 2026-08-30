# GaukukCuda

**GaukukCuda** is a GPU based hydrodynamic solver using Godunov's scheme. It is the Cuda version of **Gaukuk**

---

##  Quick Start

```bash
# 1. edit the config.mk for configuration then Compile
make -j

# 2. Run
cd bin
./gaukuk.sim -i ../input/kh.in
```

Plot the result:

```bash
cd pypkg
python plot_kh.py
```