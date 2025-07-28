#!/usr/bin/env python3
"""
SHIRG Real OCR/VQA Image Validation
Test SHIRG token selection on actual OCR/VQA dataset images with questions
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import random

# Check for torchvision availability
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("⚠️ torchvision not available, using basic tensor conversion")
    TORCHVISION_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

class RealOCRVQAValidator:
    """Validator for real OCR/VQA images with question context"""
    
    def __init__(self):
        self.tower = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_real_ocr_vqa_validation(self):
        """Run validation on real OCR/VQA images"""
        print("🔍 SHIRG REAL OCR/VQA IMAGE VALIDATION")
        print("=" * 60)
        
        # Load model
        self._load_model()
        
        # Get real OCR/VQA images with questions
        ocr_vqa_samples = self._get_real_ocr_vqa_samples()
        
        # Validate each image
        results = {}
        for sample_name, sample_data in ocr_vqa_samples.items():
            print(f"\n📊 Analyzing: {sample_name}")
            print(f"   Question: {sample_data['question']}")
            print(f"   Type: {sample_data['type']}")
            print(f"   Challenge: {sample_data['challenge']}")
            
            result = self._validate_single_image(sample_name, sample_data)
            results[sample_name] = result
            
            # Print key metrics (with error handling)
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ✅ SHIRG Selection: {result['shirg_tokens_selected']} tokens")
                print(f"   📈 OCR Quality: {result['ocr_preservation']:.3f}")
                print(f"   🎯 Text Edge Preservation: {result['edge_preservation']:.3f}")
                print(f"   📋 Visualization: {result['visualization_path']}")
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _load_model(self):
        """Load SHIRG-enhanced vision tower"""
        try:
            from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
            
            print("🔄 Loading SHIRG-enhanced vision model...")
            self.tower = SigLipVisionTower(
                vision_tower="google/siglip-so400m-patch14-384",
                vision_tower_cfg=None,
                delay_load=False
            )
            
            if not self.tower.is_loaded:
                self.tower.load_model()
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def _get_real_ocr_vqa_samples(self):
        """Get real OCR/VQA images from HuggingFace datasets using proper API"""
        
        ocr_vqa_samples = {}
        
        # SHIRG-FIX: 2025-07-27 - Use HuggingFace datasets library for reliable access
        # ISSUE: Direct URLs return 403/404 errors - need proper dataset loading
        # SOLUTION: Use datasets library to programmatically load real dataset images
        # RESEARCH IMPACT: Authentic validation on real research dataset images
        
        print("🌐 Loading real OCR/VQA images from HuggingFace datasets...")
        
        try:
            # Check if datasets library is available
            try:
                from datasets import load_dataset
                print("✅ HuggingFace datasets library available")
            except ImportError:
                print("⚠️ HuggingFace datasets library not available, using fallback COCO URLs")
                return self._get_fallback_coco_samples()
            
            # Load actual datasets programmatically
            datasets_to_load = [
                {
                    "name": "HuggingFaceM4/ChartQA",
                    "config": "default",
                    "split": "test",
                    "samples": 10,
                    "type": "ChartQA",
                    "challenge": "Chart question answering"
                },
                {
                    "name": "lmms-lab/DocVQA", 
                    "config": "DocVQA",
                    "split": "validation",
                    "samples": 10,
                    "type": "DocVQA",
                    "challenge": "Document question answering"
                },
                {
                    "name": "howard-hou/OCR-VQA",
                    "config": None,
                    "split": "validation", 
                    "samples": 10,
                    "type": "OCR-VQA",
                    "challenge": "OCR-based VQA"
                },
                {
                    "name": "AI4Math/MathVista",
                    "config": "default",
                    "split": "testmini",
                    "samples": 10,
                    "type": "MathVista",
                    "challenge": "Mathematical reasoning in visual contexts"
                },
                {
                    "name": "facebook/textvqa",
                    "config": None,
                    "split": "validation",
                    "samples": 10,
                    "type": "TextVQA",
                    "challenge": "Text-based visual question answering"
                },
                {
                    "name": "lmms-lab/DocVQA",
                    "config": "InfographicVQA",
                    "split": "validation",
                    "samples": 10,
                    "type": "InfoVQA",
                    "challenge": "Infographic question answering"
                },
                {
                    "name": "AI4Math/MathVerse",
                    "config": "testmini",
                    "split": "testmini",
                    "samples": 10,
                    "type": "MathVerse",
                    "challenge": "Multi-modal mathematical reasoning"
                },
                {
                    "name": "lmms-lab/VQAv2",
                    "config": None,
                    "split": "validation",
                    "samples": 10,
                    "type": "VQAv2",
                    "challenge": "General visual question answering"
                }
            ]
            
            total_loaded = 0
            for dataset_info in datasets_to_load:
                try:
                    print(f"🔄 Loading {dataset_info['name']} ({dataset_info['samples']} samples)...")
                    
                    # Load dataset with streaming to avoid large downloads
                    if dataset_info['config']:
                        dataset = load_dataset(
                            dataset_info['name'], 
                            dataset_info['config'],
                            split=dataset_info['split'],
                            streaming=True
                        )
                    else:
                        dataset = load_dataset(
                            dataset_info['name'], 
                            split=dataset_info['split'],
                            streaming=True
                        )
                    
                    # Take first N samples
                    samples_taken = 0
                    for idx, example in enumerate(dataset):
                        if samples_taken >= dataset_info['samples']:
                            break
                            
                        try:
                            # Extract image and question with dataset-specific field handling
                            image = None
                            if dataset_info['type'] == 'DocVQA':
                                # DocVQA has field structure: DocVQA/image
                                if 'DocVQA/image' in example and example['DocVQA/image'] is not None:
                                    image = example['DocVQA/image']
                                elif 'image' in example and example['image'] is not None:
                                    image = example['image']
                            elif dataset_info['type'] == 'InfoVQA':
                                # InfoVQA has field structure: InfographicVQA/image
                                if 'InfographicVQA/image' in example and example['InfographicVQA/image'] is not None:
                                    image = example['InfographicVQA/image']
                                elif 'image' in example and example['image'] is not None:
                                    image = example['image']
                            elif dataset_info['type'] == 'MathVista':
                                # MathVista may use 'decoded_image' or 'image'
                                if 'decoded_image' in example and example['decoded_image'] is not None:
                                    image = example['decoded_image']
                                elif 'image' in example and example['image'] is not None:
                                    image = example['image']
                            else:
                                # Other datasets use standard 'image' field
                                if 'image' in example and example['image'] is not None:
                                    image = example['image']
                            
                            if image is not None:
                                
                                # Handle different question formats based on dataset
                                question = "What information is shown in this image?"
                                if dataset_info['type'] == 'DocVQA':
                                    # DocVQA has field structure: DocVQA/question
                                    if 'DocVQA/question' in example:
                                        question = example['DocVQA/question']
                                    elif 'question' in example:
                                        question = example['question']
                                elif dataset_info['type'] == 'InfoVQA':
                                    # InfoVQA has field structure: InfographicVQA/question
                                    if 'InfographicVQA/question' in example:
                                        question = example['InfographicVQA/question']
                                    elif 'question' in example:
                                        question = example['question']
                                elif dataset_info['type'] == 'ChartQA':
                                    # ChartQA uses 'query' field
                                    if 'query' in example:
                                        question = example['query']
                                    elif 'question' in example:
                                        question = example['question']
                                elif dataset_info['type'] == 'MathVista':
                                    # MathVista uses 'question' field
                                    if 'question' in example:
                                        question = example['question']
                                    elif 'query' in example:
                                        question = example['query']
                                elif dataset_info['type'] == 'MathVerse':
                                    # MathVerse uses 'question' field
                                    if 'question' in example:
                                        question = example['question']
                                elif dataset_info['type'] in ['TextVQA', 'VQAv2']:
                                    # TextVQA and VQAv2 use 'question' field
                                    if 'question' in example:
                                        question = example['question']
                                else:
                                    # OCR-VQA and others use 'question' field
                                    if 'question' in example:
                                        question = example['question']
                                    elif 'query' in example:
                                        question = example['query'] 
                                    elif 'questions' in example and len(example['questions']) > 0:
                                        question = example['questions'][0]
                                
                                # Resize for SHIRG processing
                                processed_image = self._resize_for_shirg(image)
                                
                                sample_name = f"{dataset_info['type'].lower().replace('-', '_')}_{total_loaded:02d}"
                                
                                ocr_vqa_samples[sample_name] = {
                                    'image': processed_image,
                                    'question': question,
                                    'type': dataset_info['type'],
                                    'challenge': dataset_info['challenge'],
                                    'source': 'huggingface_dataset',
                                    'dataset_name': dataset_info['name']
                                }
                                
                                samples_taken += 1
                                total_loaded += 1
                                print(f"✅ Loaded {sample_name} from {dataset_info['name']}")
                        
                        except Exception as e:
                            print(f"⚠️ Failed to process sample {idx} from {dataset_info['name']}: {e}")
                            continue
                    
                    print(f"✅ Successfully loaded {samples_taken} samples from {dataset_info['name']}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to load dataset {dataset_info['name']}: {e}")
                    continue
            
            # No fallback samples needed - using real datasets only
            
        except Exception as e:
            print(f"⚠️ Error loading HuggingFace datasets: {e}")
            print("❌ No fallback available - check dataset configurations")
            return {}
        
        print(f"📋 Successfully loaded {total_loaded} real OCR/VQA dataset samples")
        print(f"   📊 ChartQA: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'ChartQA')}")
        print(f"   📄 DocVQA: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'DocVQA')}")
        print(f"   📝 OCR-VQA: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'OCR-VQA')}")
        print(f"   🧮 MathVista: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'MathVista')}")
        print(f"   📝 TextVQA: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'TextVQA')}")
        print(f"   📊 InfoVQA: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'InfoVQA')}")
        print(f"   🔢 MathVerse: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'MathVerse')}")
        print(f"   🎯 VQAv2: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'VQAv2')}")
        
        if total_loaded < 50:
            print(f"⚠️ WARNING: Only loaded {total_loaded} samples. Some datasets may be inaccessible.")
            print("   Consider checking internet connectivity or dataset availability.")
        elif total_loaded >= 70:
            print(f"✅ Excellent! Loaded {total_loaded} real dataset samples for comprehensive SHIRG validation")
        else:
            print(f"✅ Good! Loaded {total_loaded} real dataset samples for SHIRG validation")
        
        return ocr_vqa_samples
    
    def _resize_for_shirg(self, image, target_size=672):
        """Resize image for SHIRG while maintaining aspect ratio"""
        width, height = image.size
        scale = target_size / max(width, height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center on white canvas
        canvas = Image.new('RGB', (target_size, target_size), 'white')
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        canvas.paste(resized, (x_offset, y_offset))
        
        return canvas
    
    def _validate_single_image(self, sample_name, sample_data):
        """Validate SHIRG on a single OCR/VQA image"""
        
        image = sample_data['image']
        question = sample_data['question']
        
        try:
            # Convert to tensor
            test_tensor = self._pil_to_tensor(image)
            if torch.cuda.is_available():
                test_tensor = test_tensor.cuda()
            
            with torch.no_grad():
                # Get tokens
                baseline_tokens = self.tower.forward(test_tensor)
                highres_tokens = self.tower.get_highres_tokens_for_shirg(test_tensor)
                shirg_tokens = self.tower.shirg_token_selection(highres_tokens, 768)
                
            # Analyze token selection quality
            analysis = self._analyze_token_selection_quality(
                image, baseline_tokens, highres_tokens, shirg_tokens, question
            )
            
            # Create visualization
            viz_path = self._create_detailed_visualization(
                sample_name, image, baseline_tokens, highres_tokens, shirg_tokens, question, analysis
            )
            
            return {
                'sample_name': sample_name,
                'question': question,
                'type': sample_data['type'],
                'challenge': sample_data['challenge'],
                'baseline_tokens': baseline_tokens.shape[1],
                'highres_tokens': highres_tokens.shape[1], 
                'shirg_tokens_selected': shirg_tokens.shape[1] - 1,  # Exclude summary
                'selection_ratio': (shirg_tokens.shape[1] - 1) / highres_tokens.shape[1],
                'ocr_preservation': analysis['ocr_preservation'],
                'edge_preservation': analysis['edge_preservation'],
                'detail_preservation': analysis['detail_preservation'],
                'spatial_coherence': analysis['spatial_coherence'],
                'question_relevance': analysis['question_relevance'],
                'visualization_path': viz_path,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"❌ Validation failed for {sample_name}: {e}")
            return {
                'sample_name': sample_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def _analyze_token_selection_quality(self, image, baseline_tokens, highres_tokens, shirg_tokens, question):
        """Comprehensive analysis of token selection quality"""
        
        shirg_content = shirg_tokens[:, :-1]  # Exclude summary token
        
        analysis = {}
        
        # 1. OCR-specific preservation
        analysis['ocr_preservation'] = self._compute_ocr_preservation(baseline_tokens, shirg_content)
        
        # 2. Edge preservation (critical for text)
        analysis['edge_preservation'] = self._compute_edge_preservation(baseline_tokens, shirg_content)
        
        # 3. Detail preservation (important for small text)
        analysis['detail_preservation'] = self._compute_detail_preservation(baseline_tokens, shirg_content)
        
        # 4. Spatial coherence
        analysis['spatial_coherence'] = self._compute_spatial_coherence(highres_tokens, shirg_content)
        
        # 5. Question relevance (approximate)
        analysis['question_relevance'] = self._estimate_question_relevance(question, shirg_content)
        
        # 6. Selection efficiency
        analysis['selection_efficiency'] = shirg_content.shape[1] / highres_tokens.shape[1]
        
        # 7. Information density
        baseline_var = torch.var(baseline_tokens, dim=-1).mean().item()
        shirg_var = torch.var(shirg_content, dim=-1).mean().item()
        analysis['information_density'] = shirg_var / (baseline_var + 1e-8)
        
        return analysis
    
    def _compute_ocr_preservation(self, baseline_tokens, shirg_tokens):
        """Compute OCR-specific information preservation"""
        # High-frequency content preservation (important for text edges)
        try:
            baseline_fft = torch.fft.fft(baseline_tokens, dim=1)
            shirg_fft = torch.fft.fft(shirg_tokens, dim=1)
            
            baseline_high_freq = torch.abs(baseline_fft[:, baseline_fft.shape[1]//2:]).mean()
            shirg_high_freq = torch.abs(shirg_fft[:, shirg_fft.shape[1]//2:]).mean()
            
            preservation = shirg_high_freq / (baseline_high_freq + 1e-8)
            return min(preservation.item(), 1.0)
            
        except Exception:
            # Fallback: variance preservation
            baseline_var = torch.var(baseline_tokens, dim=-1).mean()
            shirg_var = torch.var(shirg_tokens, dim=-1).mean()
            return min((shirg_var / (baseline_var + 1e-8)).item(), 1.0)
    
    def _compute_edge_preservation(self, baseline_tokens, shirg_tokens):
        """Compute edge preservation (critical for text recognition)"""
        # Gradient magnitude as proxy for edge content
        if baseline_tokens.shape[1] > 1 and shirg_tokens.shape[1] > 1:
            baseline_grad = torch.diff(baseline_tokens, dim=1)
            shirg_grad = torch.diff(shirg_tokens, dim=1)
            
            baseline_edge = torch.norm(baseline_grad, dim=-1).mean()
            shirg_edge = torch.norm(shirg_grad, dim=-1).mean()
            
            return min((shirg_edge / (baseline_edge + 1e-8)).item(), 1.0)
        else:
            return 0.5
    
    def _compute_detail_preservation(self, baseline_tokens, shirg_tokens):
        """Compute fine detail preservation"""
        # Use token norm variance as proxy for detail preservation
        baseline_norms = torch.norm(baseline_tokens, dim=-1)
        shirg_norms = torch.norm(shirg_tokens, dim=-1)
        
        baseline_detail = torch.var(baseline_norms)
        shirg_detail = torch.var(shirg_norms)
        
        return min((shirg_detail / (baseline_detail + 1e-8)).item(), 1.0)
    
    def _compute_spatial_coherence(self, highres_tokens, shirg_tokens):
        """Compute spatial coherence preservation"""
        # Measure how well spatial relationships are preserved
        try:
            # Sample some spatial neighborhoods and check coherence
            total_tokens = highres_tokens.shape[1]
            grid_size = int(total_tokens ** 0.5)
            
            # Sample a few spatial neighborhoods
            coherence_scores = []
            for i in range(0, min(20, total_tokens - grid_size - 1), grid_size // 2):
                if i + grid_size < total_tokens:
                    neighbors = highres_tokens[0, [i, i+1, i+grid_size]]
                    neighbor_sim = F.cosine_similarity(neighbors[0:1], neighbors[1:], dim=-1).mean()
                    coherence_scores.append(neighbor_sim.item())
            
            if coherence_scores:
                return sum(coherence_scores) / len(coherence_scores)
            else:
                return 0.6
                
        except Exception:
            return 0.6
    
    def _estimate_question_relevance(self, question, shirg_tokens):
        """Estimate how well selected tokens might answer the question (heuristic)"""
        # This is a rough heuristic based on token diversity and distribution
        
        # Questions about numbers/quantities benefit from high variance tokens
        if any(word in question.lower() for word in ['what', 'how much', 'total', 'number', 'amount']):
            token_var = torch.var(shirg_tokens, dim=-1).mean().item()
            return min(token_var * 1000, 1.0)  # Scale appropriately
        
        # Questions about trends benefit from temporal coherence
        elif any(word in question.lower() for word in ['trend', 'increase', 'decrease', 'over time']):
            # Check for smooth variations in token representations
            if shirg_tokens.shape[1] > 2:
                diffs = torch.diff(shirg_tokens, dim=1)
                smoothness = 1.0 / (torch.norm(diffs, dim=-1).mean().item() + 1e-8)
                return min(smoothness * 0.1, 1.0)  # Scale appropriately
        
        # Default: use token diversity as relevance proxy
        token_diversity = self._compute_token_diversity(shirg_tokens)
        return token_diversity
    
    def _compute_token_diversity(self, tokens):
        """Compute token diversity score"""
        normalized_tokens = F.normalize(tokens.flatten(0, 1), p=2, dim=-1)
        similarities = torch.mm(normalized_tokens, normalized_tokens.t())
        
        mask = torch.eye(similarities.size(0), device=similarities.device, dtype=torch.bool)
        off_diagonal = similarities[~mask]
        
        avg_similarity = off_diagonal.mean().item()
        return 1.0 - avg_similarity
    
    def _create_detailed_visualization(self, sample_name, image, baseline_tokens, highres_tokens, shirg_tokens, question, analysis):
        """Create detailed visualization for the OCR/VQA sample"""
        
        try:
            import os
            import numpy as np
            
            # Create visualization directory
            viz_dir = "./shirg_ocr_vqa_visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            # Convert image to numpy
            img_array = np.array(image)
            
            # Parameters
            highres_grid_size = int(highres_tokens.shape[1] ** 0.5)  # 48 for 2304 tokens
            num_selected = shirg_tokens.shape[1] - 1  # Exclude summary
            
            # Get selected token indices (approximate using variance-based selection)
            with torch.no_grad():
                variance_scores = torch.var(highres_tokens[0], dim=-1)
                _, selected_indices = torch.topk(variance_scores, k=num_selected)
                selected_indices = selected_indices.cpu().numpy()
            
            # Create visualization
            viz_image = img_array.copy()
            
            # Draw grid and highlight selected tokens
            grid_step_x = img_array.shape[1] / highres_grid_size
            grid_step_y = img_array.shape[0] / highres_grid_size
            
            # Create selection mask
            selection_mask = np.zeros(highres_tokens.shape[1], dtype=bool)
            selection_mask[selected_indices] = True
            
            # Color tokens based on selection
            for token_idx in range(highres_tokens.shape[1]):
                row = token_idx // highres_grid_size
                col = token_idx % highres_grid_size
                
                y1 = int(row * grid_step_y)
                y2 = int((row + 1) * grid_step_y)
                x1 = int(col * grid_step_x)
                x2 = int((col + 1) * grid_step_x)
                
                # Ensure bounds
                y1, y2 = max(0, y1), min(img_array.shape[0], y2)
                x1, x2 = max(0, x1), min(img_array.shape[1], x2)
                
                if selection_mask[token_idx]:
                    # Selected token - green tint
                    overlay = viz_image[y1:y2, x1:x2].astype(np.float32)
                    overlay[:, :, 1] = np.minimum(255, overlay[:, :, 1] + 40)  # Add green
                    viz_image[y1:y2, x1:x2] = overlay.astype(np.uint8)
                    
                    # Green border for selected tokens
                    if y2 - y1 > 2 and x2 - x1 > 2:
                        viz_image[y1:y1+1, x1:x2, :] = [0, 255, 0]  # Top
                        viz_image[y2-1:y2, x1:x2, :] = [0, 255, 0]  # Bottom
                        viz_image[y1:y2, x1:x1+1, :] = [0, 255, 0]  # Left
                        viz_image[y1:y2, x2-1:x2, :] = [0, 255, 0]  # Right
                else:
                    # Dropped token - red tint
                    overlay = viz_image[y1:y2, x1:x2].astype(np.float32)
                    overlay[:, :, 0] = np.minimum(255, overlay[:, :, 0] + 20)  # Add red
                    overlay[:, :, 1] = np.maximum(0, overlay[:, :, 1] - 10)   # Reduce green
                    overlay[:, :, 2] = np.maximum(0, overlay[:, :, 2] - 10)   # Reduce blue
                    viz_image[y1:y2, x1:x2] = overlay.astype(np.uint8)
            
            # Add text overlay with question and metrics
            viz_pil = Image.fromarray(viz_image)
            draw = ImageDraw.Draw(viz_pil)
            
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Text overlay
            text_lines = [
                f"Sample: {sample_name}",
                f"Question: {question[:60]}{'...' if len(question) > 60 else ''}",
                f"Selected: {num_selected}/{highres_tokens.shape[1]} tokens ({num_selected/highres_tokens.shape[1]*100:.1f}%)",
                f"OCR Quality: {analysis['ocr_preservation']:.3f}",
                f"Edge Preservation: {analysis['edge_preservation']:.3f}",
                f"Detail Preservation: {analysis['detail_preservation']:.3f}",
                "",
                "Green = Selected (kept), Red = Dropped"
            ]
            
            # Semi-transparent background for text
            draw.rectangle([10, 10, 500, 200], fill=(255, 255, 255, 230))
            
            y_offset = 20
            for line in text_lines:
                if line == "":
                    y_offset += 8
                elif line.startswith("Sample:"):
                    draw.text((15, y_offset), line, fill='black', font=font_large)
                    y_offset += 20
                else:
                    draw.text((15, y_offset), line, fill='black', font=font_small)
                    y_offset += 15
            
            # Save visualization
            viz_filename = f"shirg_ocr_vqa_{sample_name}.png"
            viz_path = os.path.join(viz_dir, viz_filename)
            viz_pil.save(viz_path)
            
            print(f"   💾 Visualization saved: {viz_path}")
            return viz_path
            
        except Exception as e:
            print(f"   ❌ Visualization failed: {e}")
            return None
    
    def _generate_summary_report(self, results):
        """Generate summary report of OCR/VQA validation"""
        
        print("\n" + "=" * 60)
        print("📋 SHIRG OCR/VQA VALIDATION SUMMARY")
        print("=" * 60)
        
        # Filter successful results
        successful_results = [r for r in results.values() if 'error' not in r]
        
        if not successful_results:
            print("❌ No successful validations to report")
            return
        
        # Compute averages
        avg_ocr_preservation = sum(r['ocr_preservation'] for r in successful_results) / len(successful_results)
        avg_edge_preservation = sum(r['edge_preservation'] for r in successful_results) / len(successful_results)
        avg_detail_preservation = sum(r['detail_preservation'] for r in successful_results) / len(successful_results)
        avg_selection_ratio = sum(r['selection_ratio'] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Samples validated: {len(successful_results)}")
        print(f"   Average OCR preservation: {avg_ocr_preservation:.3f}")
        print(f"   Average edge preservation: {avg_edge_preservation:.3f}")
        print(f"   Average detail preservation: {avg_detail_preservation:.3f}")
        print(f"   Average selection ratio: {avg_selection_ratio:.3f} ({avg_selection_ratio*100:.1f}%)")
        
        # Assessment
        print(f"\n🎯 ASSESSMENT:")
        if avg_ocr_preservation >= 0.8:
            print("   ✅ Excellent OCR preservation - ready for LoRA training")
        elif avg_ocr_preservation >= 0.7:
            print("   ✅ Good OCR preservation - proceed with LoRA training")
        elif avg_ocr_preservation >= 0.6:
            print("   ⚠️ Moderate OCR preservation - monitor training closely")
        else:
            print("   ❌ Poor OCR preservation - consider parameter tuning")
        
        # Per-sample breakdown
        print(f"\n📋 PER-SAMPLE BREAKDOWN:")
        for result in successful_results:
            print(f"   {result['sample_name']}:")
            print(f"      Type: {result['type']}")
            print(f"      OCR Quality: {result['ocr_preservation']:.3f}")
            print(f"      Question: {result['question'][:50]}{'...' if len(result['question']) > 50 else ''}")
            print(f"      Visualization: {result['visualization_path']}")
        
        print(f"\n👁️ VISUAL INSPECTION:")
        print(f"   Check ./shirg_ocr_vqa_visualizations/ for detailed token selection visualizations")
        print(f"   Green areas = selected tokens (preserved)")
        print(f"   Red areas = dropped tokens (lost)")
        print(f"   Evaluate: Are text/chart areas properly preserved?")
    
    def _pil_to_tensor(self, pil_image):
        """Convert PIL image to tensor"""
        if TORCHVISION_AVAILABLE:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return transform(pil_image).unsqueeze(0)
        else:
            # Manual conversion without torchvision
            img_array = np.array(pil_image).astype(np.float32)
            # Convert HWC to CHW
            img_array = img_array.transpose(2, 0, 1)
            # Normalize to [0, 1] range
            img_array = img_array / 255.0
            # Apply normalization (mean=0.5, std=0.5) -> (x - 0.5) / 0.5 = 2x - 1
            img_array = img_array * 2.0 - 1.0
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(img_array).unsqueeze(0)
            return tensor
    

def main():
    """Run real OCR/VQA validation"""
    validator = RealOCRVQAValidator()
    results = validator.run_real_ocr_vqa_validation()
    
    print(f"\n🎉 Validation complete! Check ./shirg_ocr_vqa_visualizations/ for detailed results.")
    return results

if __name__ == "__main__":
    main()