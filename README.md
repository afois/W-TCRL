# Weight-Temporally Coded Representation Learning (W-TCRL)

This repository contains the official implementation of the method introduced in:

**[Enhanced representation learning with temporal coding in sparsely spiking neural networks](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1250908/full)**  

---

## 🔧 Train a new model

Run:

```bash
python3 launch.py
````

Training logs and model snapshots are stored as `.pickle` files.

---

## 📦 Datasets

The experiments rely on publicly available datasets:

* **MNIST** — [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* **Natural Images (Olshausen & Field)** — [http://www.rctn.org/bruno/sparsenet/](http://www.rctn.org/bruno/sparsenet/)

---

## 🧠 Learned Filters (0 → 60k iterations)

![Receptive fields](/images/mnist_cv.png)

---

## 📄 Citation

```bibtex
@article{fois2023enhanced,
  title   = {Enhanced representation learning with temporal coding in sparsely spiking neural networks},
  author  = {Fois, Adrien and Girau, Bernard},
  journal = {Frontiers in Computational Neuroscience},
  volume  = {17},
  pages   = {1250908},
  year    = {2023},
  doi     = {10.3389/fncom.2023.1250908}
}
```
