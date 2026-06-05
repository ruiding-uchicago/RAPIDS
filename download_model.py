#!/usr/bin/env python3
"""
Download UMA model from HuggingFace
Run this once after getting access to the model
"""

import sys
from getpass import getpass

try:
    from huggingface_hub import login
    from fairchem.core import pretrained_mlip
except ImportError:
    print("Please install required packages first:")
    print("pip install huggingface_hub fairchem-core")
    sys.exit(1)

print("="*60)
print("UMA Model Download Script")
print("="*60)
print("\nFirst, get your HuggingFace token:")
print("1. Register at https://huggingface.co")
print("2. Request access at https://huggingface.co/facebook/UMA")
print("3. Get token from https://huggingface.co/settings/tokens")
print()

# Get token securely
token = getpass("Enter your HuggingFace token (hidden): ")

try:
    print("\nLogging in to HuggingFace...")
    login(token=token)
    
    print("Downloading UMA-S-1p1 model (this may take a few minutes)...")
    model = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    
    print("\n✓ Model downloaded successfully!")
    print("You can now run simulations with smart_fairchem_flow.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have access to the UMA model")
    print("2. Check your token is correct")
    print("3. Ensure you have internet connection")
    sys.exit(1)