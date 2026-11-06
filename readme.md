# ANN-Assisted Multi-Objective Optimization of Sustainable Concrete Mix Design Using NSGA-II

## Project Overview

This project explores data-driven optimization of sustainable concrete mix designs by integrating Artificial Neural Networks (ANN) with the Non-Dominated Sorting Genetic Algorithm II (NSGA-II).

The objective is to develop an ANN-based predictive model for estimating compressive strength and energy consumption, followed by multi-objective optimization to identify Pareto-optimal concrete mix designs.

## Methodology

1. **Dataset Analysis:** Exploratory data analysis and identification of relationships between concrete mix parameters, compressive strength and energy consumption.
2. **Data Preprocessing:** Feature selection, Min-Max normalization and training, validation and testing data partitioning.
3. **ANN Modelling:** Development of a multi-layer neural network using TensorFlow/Keras with ReLU activation, dropout regularization and early stopping.
4. **Model Evaluation:** Performance assessment using Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R².
5. **Multi-Objective Optimization:** Integration of the ANN surrogate model with NSGA-II using pymoo.
6. **Pareto Analysis:** Identification of non-dominated solutions representing trade-offs between compressive strength and energy consumption.

## Optimization Objectives

- Maximize compressive strength.
- Minimize energy consumption.

## Technology Stack

Python | TensorFlow | Keras | NumPy | Pandas | Scikit-learn | Matplotlib | pymoo | NSGA-II

## Key Outcomes

- ANN-based prediction of concrete engineering properties.
- Multi-objective optimization of concrete mix designs.
- Pareto-front visualization for evaluating competing engineering objectives.
- Data-driven insights supporting energy-efficient material design.

## Repository Contents

- `main.py` – Python implementation.
- `Optimization_PPT.ipynb` – Original Jupyter Notebook.
- `data/` – Input dataset.
- `outputs/` – Generated plots and optimization results.
- `requirements.txt` – Python dependencies.

## Academic Context

Optimization Techniques – M.Tech Chemical Engineering

Indian Institute of Technology Hyderabad