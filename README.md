# Forschungspraxis: VAE-Based Indoor Localization

## Introduction

This project applies a Variational Autoencoder (VAE) for fingerprinting-based indoor localization, focusing on using the latent representation of the signal's multipath component for the localization task. The goal is to achieve accurate indoor positioning by analyzing the complex behavior of signal propagation within closed environments.

## Structure

- `scenarios/`: Folder containing the different indoor scenarios, their signal data and indoor geometry information.
- `trained_models/`: Contains checkpoints of trained models for evaluation.
- `utils/`: Contains utility functions and classes.
- `vae_model.py`: VAE Implementation and training.
- `evaluate_model.py`: Evaluates the trained model and visualizes the latent representation and reconstruction of fingerprints and testing points.
