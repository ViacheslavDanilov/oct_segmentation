#!/bin/bash

# List of cam_methods to iterate over
cam_methods=("GradCAM" "HiResCAM" "GradCAMElementWise" "GradCAMPlusPlus" "XGradCAM" "AblationCAM" "EigenCAM" "EigenGradCAM" "LayerCAM")

# Iterate over each cam_method
for method in "${cam_methods[@]}"
do
    echo "Running visualize_activation_maps with cam_method: $method"
    python src/models/visualize_activation_maps.py cam_method="$method"
done
