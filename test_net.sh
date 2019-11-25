#!/bin/bash

# Image and model names
TEST_IMG=ADE_val_00001519.jpg
MODEL_PATH=TEST/trained_networks

INPUT_PATH=TEST/input_images
OUTPUT_PATH=TEST/output_images

# Inference
python -u test.py \
  --model_path $MODEL_PATH \
  --test_imgs $INPUT_PATH \
  --arch_encoder resnet50dilated \
  --arch_decoder ppm_deepsup \
  --fc_dim 2048 \
  --result $OUTPUT_PATH \
  --num_class 5
