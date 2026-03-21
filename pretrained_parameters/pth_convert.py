#!/usr/bin/env python3
"""
PyTorch ResNet Model to SEAL CNN Format Converter

This script converts a PyTorch .pth model to the txt format required by the MPCNN project.
It handles ResNet models for CIFAR-10/CIFAR-100 datasets.

Usage:
    python convert_pth_to_seal.py --model resnet110 --dataset cifar10 --input model.pth --output converted_model/

The script will:
1. Load the PyTorch model
2. Extract and transform parameters to the required format
3. Validate the conversion by checking shapes
4. Save parameters as txt files
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class ResNetCIFAR(nn.Module):
    """
    ResNet for CIFAR-like datasets (32x32 images).
    This is a reference model structure to understand parameter names.
    """
    def __init__(self, num_classes=10, block='BasicBlock', layers=[18, 18, 18]):
        super(ResNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 16, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)


def convert_resnet_to_seal_format(model_path, output_dir, model_name='resnet20', dataset='cifar10'):
    """
    Convert a PyTorch ResNet model to SEAL CNN txt format.
    
    Args:
        model_path: Path to the .pth model file
        output_dir: Directory to save converted txt files
        model_name: Model name (resnet20, resnet32, resnet44, resnet56, resnet20)
        dataset: Dataset name (cifar10 or cifar100)
    
    Returns:
        bool: True if conversion successful
    """
    
    print(f"Loading model from {model_path}...")
    
    # Determine layer configuration based on model name
    layer_configs = {
        'resnet20': ([3, 3, 3], 16, 10),
        'resnet32': ([5, 5, 5], 16, 10),
        'resnet44': ([7, 7, 7], 16, 10),
        'resnet56': ([9, 9, 9], 16, 10),
        'resnet110': ([18, 18, 18], 16, 10),
    }
    
    if model_name not in layer_configs:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(layer_configs.keys())}")
    
    layers, initial_channels, num_classes = layer_configs[model_name]
    if dataset == 'cifar100':
        num_classes = 100
    
    # Load state dict
    if model_path.endswith('.pth') or model_path.endswith('.pt'):
        state_dict = torch.load(model_path, map_location='cpu')
        print(state_dict.keys())
        if 'net' in state_dict.keys():
            state_dict = state_dict['net']
        # Handle wrapped models (with 'module.' prefix)
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("Detected DataParallel wrapped model, removing 'module.' prefix...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        raise ValueError(f"Unsupported file format: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Model loaded successfully. Converting to SEAL format...")
    
    # Extract parameters
    converted_params = {}
    
    # 1. conv1 weight (first convolutional layer)
    # Shape: [out_channels, in_channels, kernel_h, kernel_w] -> [fh, fw, ci, co]
    print(state_dict.keys())
    conv1_weight = state_dict['conv1.weight'].numpy()
    # Expected shape: [16, 3, 3, 3] for CIFAR
    print(f"conv1.weight shape: {conv1_weight.shape}")
    # Transform: [co, ci, fh, fw] -> [fh, fw, ci, co]
    # conv1_weight_seal = np.transpose(conv1_weight, (2, 3, 1, 0))
    conv1_weight_seal = conv1_weight
    converted_params['conv1_weight'] = conv1_weight_seal.flatten()
    
    # 2. bn1 parameters (first batch normalization)
    bn1_bias = state_dict['bn1.bias'].numpy()
    bn1_running_mean = state_dict['bn1.running_mean'].numpy()
    bn1_running_var = state_dict['bn1.running_var'].numpy()
    bn1_weight = state_dict['bn1.weight'].numpy()
    
    print(f"bn1 shapes - bias: {bn1_bias.shape}, mean: {bn1_running_mean.shape}, var: {bn1_running_var.shape}, weight: {bn1_weight.shape}")
    
    converted_params['bn1_bias'] = bn1_bias
    converted_params['bn1_running_mean'] = bn1_running_mean
    converted_params['bn1_running_var'] = bn1_running_var
    converted_params['bn1_weight'] = bn1_weight
    converted_params['bn1_num_batches_tracked'] = np.array([0])
    
    # 3. Process residual layers
    # For resnet110: layer1 has 18 blocks, layer2 has 18 blocks, layer3 has 18 blocks
    # But in the original code, end_num for resnet110 is 17, meaning blocks 0-17 (18 total)
    
    layer_channels = [16, 32, 64]  # channel progression
    stage_configs = [
        (1, 16, 16),    # layer1: in_ch=16, out_ch=16
        (2, 16, 32),    # layer2: in_ch=16, out_ch=32
        (3, 32, 64),    # layer3: in_ch=32, out_ch=64
    ]
    
    stage_idx = 0
    for stage_num in [1, 2, 3]:
        in_channels = stage_configs[stage_idx][1]
        out_channels = stage_configs[stage_idx][2]
        num_blocks = layers[stage_idx]
        
        # Determine which blocks have shortcut (stride != 1)
        # For layer2 and layer3, first block has stride=2 (shortcut)
        # For resnet110/layers=[18,18,18], blocks 0-17 in each stage
        
        for block_idx in range(num_blocks):
            block_prefix = f'layer{stage_num}.{block_idx}.'
            
            # Determine if this block uses downsampling
            # In standard ResNet, first block of layer2 and layer3 has stride=2
            is_first_block_of_stage = (block_idx == 0)
            has_downsample = (stage_num > 1) and is_first_block_of_stage
            
            if has_downsample:
                # This block has a shortcut connection
                shortcut_prefix = block_prefix + 'shortcut.'
                
                # shortcut.0 is conv with stride 2
                if (shortcut_prefix + 'conv.weight') in state_dict:
                    shortcut_weight = state_dict[shortcut_prefix + 'conv.weight'].numpy()
                    # shortcut_weight_seal = np.transpose(shortcut_weight, (2, 3, 1, 0))
                    shortcut_weight_seal = shortcut_weight
                    key = f'layer{stage_num}_{block_idx}_shortcut_0_weight'
                    converted_params[key] = shortcut_weight_seal.flatten()
                    print(f"{key} shape: {shortcut_weight_seal.shape}")
                
                # shortcut.1 is bn
                for bn_suffix in ['bias', 'running_mean', 'running_var', 'weight', 'num_batches_tracked']:
                    if bn_suffix == 'num_batches_tracked':
                        # Inference value is 0
                        converted_params[f'layer{stage_num}_{block_idx}_shortcut_1_num_batches_tracked'] = np.array([0])
                    else:
                        if (shortcut_prefix + f'bn.{bn_suffix}') in state_dict:
                            param_name = f'layer{stage_num}_{block_idx}_shortcut_1_{bn_suffix}'
                            converted_params[param_name] = state_dict[shortcut_prefix + f'bn.{bn_suffix}'].numpy()
                            print(f"{param_name} shape: {converted_params[param_name].shape}")
                            
            # Process the two convolutions in each block
            for conv_idx in [1, 2]:
                conv_prefix = block_prefix + f'conv{conv_idx}.'
                bn_prefix = block_prefix + f'bn{conv_idx}'
                
                # Get input channels for this conv
                # conv1 of first block in stage uses previous stage's output channels
                # conv1 of other blocks uses out_channels
                # conv2 always uses out_channels
                
                if conv_idx == 1:
                    if is_first_block_of_stage:
                        current_in_channels = in_channels
                    else:
                        current_in_channels = out_channels
                else:
                    current_in_channels = out_channels
                
                current_out_channels = out_channels
                
                # Convolution weight
                if (conv_prefix + 'weight') in state_dict:
                    conv_weight = state_dict[conv_prefix + 'weight'].numpy()
                    # Shape: [out_ch, in_ch, kh, kw] -> [kh, kw, in_ch, out_ch]
                    # conv_weight_seal = np.transpose(conv_weight, (2, 3, 1, 0))
                    conv_weight_seal = conv_weight
                    # Validate shape
                    expected_shape = (3, 3, current_in_channels, current_out_channels)
                    if conv_weight_seal.shape != expected_shape:
                        print(f"WARNING: {conv_prefix}weight shape mismatch. Expected {expected_shape}, got {conv_weight_seal.shape}")
                    
                    key = f'layer{stage_num}_{block_idx}_conv{conv_idx}_weight'
                    converted_params[key] = conv_weight_seal.flatten()
                    print(f"{key} shape: {conv_weight_seal.shape}")
                
                # BatchNorm parameters
                for bn_param in ['bias', 'running_mean', 'running_var', 'weight', 'num_batches_tracked']:
                    if bn_param == 'num_batches_tracked':
                        # Inference value is 0
                        param_name = f'layer{stage_num}_{block_idx}_bn{conv_idx}_num_batches_tracked'
                        converted_params[param_name] = np.array([0])
                        print(f"{param_name} added (value 0)")
                    else:
                        full_key = bn_prefix + '.' + bn_param
                        if full_key in state_dict:
                            param_name = f'layer{stage_num}_{block_idx}_bn{conv_idx}_{bn_param}'
                            converted_params[param_name] = state_dict[full_key].numpy()
                            print(f"{param_name} shape: {converted_params[param_name].shape}")
        
        stage_idx += 1
    
    # 4. Final fully connected layer
    fc_weight = state_dict['fc.weight'].numpy()
    fc_bias = state_dict['fc.bias'].numpy()
    
    print(f"fc.weight shape: {fc_weight.shape}")  # [num_classes, 64]
    print(f"fc.bias shape: {fc_bias.shape}")      # [num_classes]
    
    # Transform: [out_features, in_features] -> [in_features, out_features] for consistency
    # fc_weight_seal = np.transpose(fc_weight, (1, 0))
    fc_weight_seal = fc_weight
    converted_params['linear_weight'] = fc_weight_seal.flatten()
    converted_params['linear_bias'] = fc_bias
    
    # Save all parameters to txt files
    print(f"\nSaving converted parameters to {output_dir}...")
    
    for param_name, param_data in converted_params.items():
        filepath = os.path.join(output_dir, f"{param_name}.txt")
        
        # Handle both 1D and higher dimensional arrays
        if param_data.ndim > 1:
            param_data = param_data.flatten()
        
        # Save as space-separated values (each value on new line for 1D, or space for higherD)
        with open(filepath, 'w') as f:
            for val in param_data:
                f.write(f"{val:.6e}\n")
        
        print(f"Saved {param_name} to {filepath} ({len(param_data)} values)")
    
    # Generate summary
    # summary_path = os.path.join(output_dir, 'conversion_summary.txt')
    # with open(summary_path, 'w') as f:
    #     f.write(f"Model Conversion Summary\n")
    #     f.write(f"=" * 50 + "\n")
    #     f.write(f"Original model: {model_path}\n")
    #     f.write(f"Model architecture: {model_name}\n")
    #     f.write(f"Dataset: {dataset}\n")
    #     f.write(f"Number of parameters: {len(converted_params)}\n")
    #     f.write(f"\nParameter files generated:\n")
    #     for param_name, param_data in converted_params.items():
    #         f.write(f"  {param_name}.txt: {len(param_data)} values\n")
    
    print(f"\nConversion completed successfully!")
    # print(f"Summary saved to: {summary_path}")
    
    # Validate by comparing with expected parameter counts
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)
    
    # Expected parameter count for resnet110 on cifar10:
    # - conv1: 3*3*3*16 = 432
    # - bn1: 16 * 4 = 64
    # - layer1: 18 blocks * 2 convs * (3*3*16*16 + 16*4) = 18 * 2 * (2304 + 64) = 18 * 2 * 2368 = 85248
    # - layer2: 18 blocks * 2 convs * ...
    # - layer3: 18 blocks * 2 convs * ...
    # - fc: 10 * 64 = 640
    
    print("\nExpected shapes for resnet110 on cifar10:")
    print("  conv1_weight: (3, 3, 3, 16) = 432 values")
    print("  bn1_bias, bn1_running_mean, bn1_running_var, bn1_weight: (16,) = 16 values each")
    print("  linear_weight: (64, 10) = 640 values")
    print("  linear_bias: (10,) = 10 values")
    
    return True


def validate_conversion(original_pth_path, converted_dir, model_name='resnet110'):
    """
    Validate the conversion by checking parameter shapes and values.
    """
    print("\n" + "=" * 50)
    print("VALIDATION CHECK")
    print("=" * 50)
    
    # Load original model
    original_state = torch.load(original_pth_path, map_location='cpu')
    if any(k.startswith('module.') for k in original_state.keys()):
        original_state = {k.replace('module.', ''): v for k, v in original_state.items()}
    
    # Check key parameters exist
    key_params = ['conv1.weight', 'bn1.weight', 'fc.weight']
    for param in key_params:
        if param not in original_state:
            print(f"ERROR: Missing parameter {param} in original model")
            return False
    
    # Load and check a converted file
    test_file = os.path.join(converted_dir, 'conv1_weight.txt')
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            lines = f.readlines()
        print(f"conv1_weight.txt contains {len(lines)} values")
        print(f"Expected: 3*3*3*16 = 432 values")
        if len(lines) == 432:
            print("✓ Shape check passed!")
        else:
            print("✗ Shape check failed!")
    
    print("\nValidation completed.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch ResNet model to SEAL CNN format')
    parser.add_argument('--model', type=str, default='resnet110', 
                        choices=['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10'],
                        help='Dataset type')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .pth model file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for converted txt files')
    parser.add_argument('--validate', action='store_true',
                        help='Validate conversion after conversion')
    
    args = parser.parse_args()
    
    # Convert model
    success = convert_resnet_to_seal_format(
        model_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        dataset=args.dataset
    )
    
    if success and args.validate:
        validate_conversion(args.input, args.output, args.model)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())