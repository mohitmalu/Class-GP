# Class-GP: Modeling Non-Stationary Functions with Partitioned Gaussian Processes

Gaussian Processes (GPs) are a powerful and widely used framework for modeling expensive black-box functions. They are particularly effective in applications such as Bayesian optimization, control systems, and scientific machine learning. However, standard GP models typically rely on **stationary kernels**, which assume that statistical properties of the function are consistent across the entire input space.

In many real-world scenarios—such as **cyber-physical systems, control, and safety-critical optimization**—this assumption does not hold. These systems often exhibit **local stationarity but global non-stationarity**, where different regions of the input space follow different functional behaviors. In such cases, standard GPs can lead to poor modeling performance.

---

## 🚀 Overview

**Class-GP (Class Gaussian Process)** is a novel modeling approach designed to handle a class of non-stationary functions by:

* Partitioning the input space into multiple sub-regions
* Assigning a **locally stationary GP model** to each region
* Activating **one GP per region** based on the input
* Learning hyperparameters for a class of GP's using ** Novel Likelihood function **

This allows Class-GP to:

* Capture complex, non-stationary behaviors
* Maintain interpretability through localized models
* Improve predictive performance over standard stationary GPs

---

## 🧠 Key Idea

We assume that the target function can be expressed as:

> A composition of locally stationary functions, each active over a specific sub-region of the input space.

Instead of fitting a single global GP, Class-GP:

1. Identifies partitions in the input space
2. Learns a stationary GP within each partition (using Novel likelihoods)
3. Selects the appropriate GP for prediction based on the input location

---

## 📂 Repository Structure

```
.
├── main_cgp.py     # Entry point: runs the Class-GP model
├── utils_cgp.py    # Utility functions (training, partitioning, GP handling)
├── plot_cgp.py     # Visualization and plotting utilities
└── README.md
```

### File Descriptions

* **`main_cgp.py`**
  Contains the main pipeline to train and evaluate the Class-GP model.

* **`utils_cgp.py`**
  Includes helper functions for:

  * Data handling
  * Partitioning logic
  * GP training and inference

* **`plot_cgp.py`**
  Provides visualization tools to:

  * Plot predictions
  * Compare model performance
  * Visualize partitions

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mohitmalu/Class-GP.git
```


## ▶️ Usage

To run the Class-GP model:

```bash
python main_cgp.py
```

You can modify parameters such as:

* Number of partitions
* Kernel type
* Training data size

(Refer to `main_cgp.py` for configurable options.)

---

## 📊 Applications

Class-GP is particularly useful in:

* Bayesian Optimization in non-stationary settings
* Control systems and robotics
* Semiconductor process optimization
* Safety-critical system modeling
* Scientific machine learning

---

## 📈 Results

Class-GP demonstrates improved performance over standard stationary GP models in:

* Non-stationary function approximation
* Predictive accuracy in heterogeneous domains
* Sample efficiency in complex optimization tasks

---

## 📚 Citation

If you find this work useful, please consider citing:

```
@inproceedings{malu2023class,
  title={Class GP: Gaussian process modeling for heterogeneous functions},
  author={Malu, Mohit and Pedrielli, Giulia and Dasarathy, Gautam and Spanias, Andreas},
  booktitle={International Conference on Learning and Intelligent Optimization},
  pages={408--423},
  year={2023},
  organization={Springer}
}
```

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!

Feel free to fork the repo and submit a pull request.

---

## 📬 Contact

For questions or collaborations, please reach out via:

* GitHub Issues
* Email: [mohitmalu21@gmail.com](mailto:mohitmalu21@gmail.com)

---

## ⭐ Acknowledgements

This work is inspired by challenges in modeling non-stationary systems in real-world applications, particularly in domains where data is sparse, noisy, and expensive to obtain.

