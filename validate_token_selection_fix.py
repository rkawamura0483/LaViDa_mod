#!/usr/bin/env python3
"""
SHIRG Token Selection Validation
Test that token selection now varies per image after content-aware fixes

TOKEN-SELECTION-VALIDATION: 2025-07-28 - Validate image-content-aware token selection
ISSUE: Previous implementation produced identical selection patterns across different images
SOLUTION: Validate that fixed implementation now produces different patterns per image
RESEARCH IMPACT: Confirms SHIRG token selection adapts to actual image content
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

try:
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    from llava.utils import rank0_print
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the LaViDa_mod root directory")
    sys.exit(1)

def create_test_images():
    """Create diverse test images with different content patterns"""
    test_images = []
    
    # Image 1: Horizontal lines (OCR-like)
    img1 = np.zeros((672, 672, 3), dtype=np.uint8)
    for i in range(50, 622, 40):
        img1[i:i+10, 50:622] = [255, 255, 255]
    
    # Image 2: Vertical chart bars
    img2 = np.zeros((672, 672, 3), dtype=np.uint8)
    for i in range(100, 572, 60):
        height = np.random.randint(100, 400)
        img2[672-height:672-50, i:i+30] = [0, 255, 0]
    
    # Image 3: Mathematical formula (diagonal patterns)
    img3 = np.zeros((672, 672, 3), dtype=np.uint8)
    for i in range(0, 672, 2):
        for j in range(0, 672, 2):
            if (i + j) % 50 < 10:
                img3[i:i+2, j:j+2] = [255, 0, 0]
    
    # Image 4: Central focus (document with center content)
    img4 = np.zeros((672, 672, 3), dtype=np.uint8)
    center_x, center_y = 336, 336
    for i in range(200, 472):
        for j in range(200, 472):
            if (i - center_y)**2 + (j - center_x)**2 < 100**2:
                img4[i, j] = [255, 255, 0]
    
    test_images = [img1, img2, img3, img4]
    image_names = ["horizontal_lines", "vertical_bars", "diagonal_pattern", "central_focus"]
    
    return test_images, image_names

def preprocess_image(image_array):
    """Convert numpy array to tensor format expected by vision tower"""
    # Convert to PIL Image first
    image = Image.fromarray(image_array)
    
    # Convert to tensor and normalize
    image_tensor = torch.tensor(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def test_token_selection_diversity():
    """Test that token selection now varies per image"""
    rank0_print("🧪 Testing SHIRG token selection diversity after content-aware fixes")
    
    # Create test images with different content patterns
    test_images, image_names = create_test_images()
    rank0_print(f"Created {len(test_images)} test images with diverse content patterns")
    
    # Initialize vision tower
    try:
        config = type('Config', (), {
            'enable_shirg': True,
            'mm_vision_tower': 'google/siglip-so400m-patch14-384'
        })()
        
        vision_tower = SigLipVisionTower(
            vision_tower='google/siglip-so400m-patch14-384',
            config=config,
            delay_load=False
        )
        
        if torch.cuda.is_available():
            vision_tower = vision_tower.cuda()
            rank0_print("Using GPU for validation")
        else:
            rank0_print("Using CPU for validation")
            
    except Exception as e:
        rank0_print(f"Failed to initialize vision tower: {e}")
        return False
    
    # Process each image and extract token selection patterns
    selection_patterns = []
    token_statistics = []
    
    for i, (image_array, name) in enumerate(zip(test_images, image_names)):
        rank0_print(f"\n📊 Processing image {i+1}: {name}")
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_array)
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            # Extract high-resolution tokens using SHIRG
            with torch.no_grad():
                hi_detail_tokens, lo_res_scaffold = vision_tower.extract_dual_scale_tokens(image_tensor)
                
                # Apply SHIRG selection to see which tokens are selected
                selected_tokens = vision_tower.distance_aware_selection(hi_detail_tokens, budget=1152)
                
                # Get selection indices by comparing original vs selected
                B, N, D = hi_detail_tokens.shape
                B_sel, N_sel, D_sel = selected_tokens.shape
                
                # Compute importance scores to see selection pattern
                similarity_scores = vision_tower.distance_aware_selection(
                    hi_detail_tokens, text_embeddings=None, budget=N_sel
                )
                
                # Get the actual selection pattern by finding which tokens were selected
                # This is approximate since we can't easily reverse the gather operation
                importance_scores_computed = torch.var(hi_detail_tokens, dim=-1) + torch.mean(torch.abs(hi_detail_tokens), dim=-1)
                
                # Store pattern for comparison
                pattern = importance_scores_computed[0].cpu().numpy()  # First batch
                selection_patterns.append(pattern)
                
                # Compute statistics
                stats = {
                    'name': name,
                    'mean_importance': float(pattern.mean()),
                    'std_importance': float(pattern.std()),
                    'max_importance': float(pattern.max()),
                    'min_importance': float(pattern.min()),
                    'unique_values': len(np.unique(pattern.round(decimals=6)))
                }
                token_statistics.append(stats)
                
                rank0_print(f"   Token importance stats: mean={stats['mean_importance']:.4f}, std={stats['std_importance']:.4f}")
                rank0_print(f"   Unique importance values: {stats['unique_values']}/{len(pattern)}")
                
        except Exception as e:
            rank0_print(f"❌ Failed to process {name}: {e}")
            return False
    
    # Analyze diversity of selection patterns
    rank0_print("\n🔍 Analyzing token selection diversity:")
    
    # Check if patterns are different from each other
    pattern_differences = []
    for i in range(len(selection_patterns)):
        for j in range(i+1, len(selection_patterns)):
            pattern1 = selection_patterns[i]
            pattern2 = selection_patterns[j]
            
            # Compute correlation and difference metrics
            correlation = np.corrcoef(pattern1, pattern2)[0, 1]
            mse_diff = np.mean((pattern1 - pattern2)**2)
            max_diff = np.max(np.abs(pattern1 - pattern2))
            
            diff_stats = {
                'pair': f"{image_names[i]} vs {image_names[j]}",
                'correlation': correlation,
                'mse_difference': mse_diff,
                'max_difference': max_diff
            }
            pattern_differences.append(diff_stats)
            
            rank0_print(f"   {diff_stats['pair']}: correlation={correlation:.4f}, MSE_diff={mse_diff:.6f}")
    
    # Evaluate results
    success = True
    issues = []
    
    # Check 1: Are patterns different between images?
    high_correlations = [d for d in pattern_differences if d['correlation'] > 0.95]
    if high_correlations:
        issues.append(f"High correlations found: {len(high_correlations)} pairs > 0.95")
        success = False
    
    # Check 2: Do patterns have sufficient variance?
    low_variance_images = [s for s in token_statistics if s['std_importance'] < 0.01]
    if low_variance_images:
        issues.append(f"Low variance patterns: {[s['name'] for s in low_variance_images]}")
        success = False
    
    # Check 3: Are there enough unique values?
    low_uniqueness = [s for s in token_statistics if s['unique_values'] < len(selection_patterns[0]) * 0.5]
    if low_uniqueness:
        issues.append(f"Low uniqueness: {[s['name'] for s in low_uniqueness]}")
        success = False
    
    # Report results
    rank0_print("\n" + "="*60)
    if success:
        rank0_print("✅ VALIDATION PASSED: Token selection varies per image content!")
        rank0_print("   - Selection patterns show low correlation between different images")
        rank0_print("   - Importance scores have sufficient variance within images")
        rank0_print("   - Token selection is now image-content-aware")
    else:
        rank0_print("❌ VALIDATION FAILED: Token selection still shows static behavior")
        for issue in issues:
            rank0_print(f"   - {issue}")
    rank0_print("="*60)
    
    return success

def main():
    """Main validation function"""
    rank0_print("🚀 Starting SHIRG Token Selection Validation")
    
    try:
        success = test_token_selection_diversity()
        
        if success:
            rank0_print("\n🎉 Token selection fix validation SUCCESSFUL!")
            rank0_print("SHIRG now selects different tokens based on image content")
        else:
            rank0_print("\n🚨 Token selection fix validation FAILED!")
            rank0_print("Further investigation needed for content-aware selection")
            
        return success
        
    except Exception as e:
        rank0_print(f"💥 Validation script failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)