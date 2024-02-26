# Forschungspraxis: VAE-Based Indoor Localization

## Introduction

This project applies a Variational Autoencoder (VAE) for fingerprinting-based indoor localization, focusing on using the latent representation of the signal's multipath component for the localization task. The goal is to achieve accurate indoor positioning by analyzing the complex behavior of signal propagation within closed environments.

## Structure

- `csv/`: Folder with csv input files to the QD simulator
- `data/`: Directory for storing datasets of different indoor localization scenarios.
- `trained_models/`: Contains weights and input dimensions of trained models for evaluation.
- `utils/`: Contains utility functions and classes.
- `train_vae.py`: Script to train model and save weights.
- `vae_model.py`: VAE Implementation.
- `visualize_latent.py`: Visualizes the latent representation of fingerprints and testing points using trained model.
