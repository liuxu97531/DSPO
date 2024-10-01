# Enhancing deep learning-based field reconstruction with differentiable learning framework
Sample codes for training of the general bi-level differentiable learning framework that seamlessly integrates reconstruction models with sensor placement optimization(DSPO).
The present framework co-optimize sensor placement and field reconstructions based on neural networks.

# Reference
Xu Liu, Wei Peng, Xiaoya Zhang, Xiaoyu Zhao, Weien Zhou, Wen Yao, and Xiaoqian Chen, "Enhancing deep learning-based field reconstruction with differentiable learning framework,", arXiv preprint arXiv:.


# Information
Author: Xu Liu
This repository contains
1. cylinder2D case:
   - DSPO_fno_cy.py: train FNO with DSPO.
   - DSPO_gnn_cy.py: train GCN with DSPO.
   - DSPO_mlp_cnn_cy.py: train GCN with DSPO.
   - DSPO_mlp_cy.py: train mlp with DSPO.
   - DSPO_podnn_cy.py: train podnn with DSPO.
   - DSPO_senseiver_cy.py: train senseiver with DSPO.
   - CylinderDataset.py: load the cylinder2D data for training.
   - sensor_ini.py: load the init placement.
2. models file: reconstruction models including CNN, MLP, PODNN, GCN, FNO, Senseiver and so on.
3. utils file: files including differential operator.

# Requirements
- python 3.X (>3.8)
- torch torch1.13.1+cu116
- numpy
- pandas
- pickle