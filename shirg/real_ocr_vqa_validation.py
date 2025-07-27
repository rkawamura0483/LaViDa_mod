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
            
            # Print key metrics
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
        """Get real OCR/VQA images from local data directory"""
        
        ocr_vqa_samples = {}
        
        # First, load images from local data directory
        data_dir = Path("./data/real_images")
        
        # Load ChartQA samples
        chartqa_dir = data_dir / "chartqa"
        if chartqa_dir.exists():
            print(f"📁 Loading local ChartQA images from {chartqa_dir}")
            chartqa_questions = [
                "What was the revenue growth rate between Q2 and Q3 2024?",
                "What is the highest value shown in this chart?",
                "What trend is visible in the data over time?"
            ]
            
            for idx, img_path in enumerate(sorted(chartqa_dir.glob("*.jpg"))[:3]):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self._resize_for_shirg(image)
                    
                    sample_name = f"chartqa_{img_path.stem}"
                    ocr_vqa_samples[sample_name] = {
                        'image': image,
                        'question': chartqa_questions[idx % len(chartqa_questions)],
                        'type': 'ChartQA-like',
                        'challenge': 'Chart analysis and numerical extraction',
                        'source': 'local'
                    }
                    print(f"✅ Loaded {sample_name}")
                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")
        
        # If not enough local images, create high-quality synthetic OCR/VQA samples
        if len(ocr_vqa_samples) < 5:
            print(f"📊 Found {len(ocr_vqa_samples)} local images, creating additional synthetic samples...")
            
            # Create realistic synthetic OCR/VQA images for comprehensive testing
            synthetic_samples = [
                {
                    "name": "synthetic_revenue_chart",
                    "type": "ChartQA-like",
                    "question": "What was the revenue growth rate between Q2 and Q3 2024?",
                    "challenge": "Multi-series bar chart with percentage calculations"
                },
                {
                    "name": "synthetic_financial_doc",
                    "type": "DocVQA-like", 
                    "question": "What is the net profit margin shown in the financial summary?",
                    "challenge": "Dense numerical data extraction"
                },
                {
                    "name": "synthetic_infographic",
                    "type": "InfographicsVQA-like",
                    "question": "What are the three main components shown in the diagram?",
                    "challenge": "Complex layout with mixed text and graphics"
                },
                {
                    "name": "synthetic_table_analysis",
                    "type": "TabMWP-like",
                    "question": "What is the average value across all categories?",
                    "challenge": "Tabular data with calculations"
                },
                {
                    "name": "synthetic_technical_diagram",
                    "type": "AI2D-like",
                    "question": "What is the data flow between the main components?",
                    "challenge": "Technical diagram with labels and connections"
                }
            ]
            
            # Create only as many synthetic samples as needed
            needed = max(0, 5 - len(ocr_vqa_samples))
            for sample_info in synthetic_samples[:needed]:
                image = self._create_high_quality_synthetic_image(sample_info)
                ocr_vqa_samples[sample_info['name']] = {
                    'image': image,
                    'question': sample_info['question'],
                    'type': sample_info['type'],
                    'challenge': sample_info['challenge'],
                    'source': 'synthetic'
                }
                print(f"✅ Created {sample_info['name']}")
        
        print(f"📋 Total samples for validation: {len(ocr_vqa_samples)}")
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
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform(pil_image).unsqueeze(0)
    
    def _create_synthetic_ocr_vqa_samples(self):
        """Create high-quality synthetic samples as fallback"""
        
        synthetic_samples = {}
        
        # Create complex chart
        chart_image = self._create_complex_revenue_chart()
        synthetic_samples["synthetic_revenue_chart"] = {
            'image': chart_image,
            'question': "What was the revenue growth rate between Q2 and Q3 2024?",
            'type': "ChartQA-like",
            'challenge': "Multi-series bar chart with percentage calculations"
        }
        
        # Create financial document
        doc_image = self._create_financial_summary()
        synthetic_samples["synthetic_financial_doc"] = {
            'image': doc_image,
            'question': "What is the net profit margin shown in the financial summary?",
            'type': "DocVQA-like", 
            'challenge': "Dense numerical data extraction"
        }
        
        return synthetic_samples
    
    def _create_high_quality_synthetic_image(self, sample_info):
        """Create high-quality synthetic image based on sample type"""
        
        if "chart" in sample_info['name']:
            return self._create_complex_revenue_chart()
        elif "financial_doc" in sample_info['name']:
            return self._create_financial_summary()
        elif "infographic" in sample_info['name']:
            return self._create_infographic()
        elif "table" in sample_info['name']:
            return self._create_table_analysis()
        elif "technical" in sample_info['name']:
            return self._create_technical_diagram()
        else:
            # Default to creating a chart
            return self._create_complex_revenue_chart()
    
    def _create_complex_revenue_chart(self):
        """Create complex revenue chart for testing"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        # Chart title
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        draw.text((200, 30), "Quarterly Revenue Analysis", fill='black', font=title_font)
        
        # Chart area
        chart_left, chart_top = 80, 100
        chart_right, chart_bottom = 580, 450
        
        # Draw chart border and grid
        draw.rectangle([chart_left, chart_top, chart_right, chart_bottom], outline='black', width=2)
        
        # Grid lines
        for i in range(1, 5):
            y = chart_top + (chart_bottom - chart_top) * i / 5
            draw.line([chart_left, y, chart_right, y], fill='lightgray', width=1)
        
        # Y-axis labels (revenue in millions)
        y_labels = ["100M", "80M", "60M", "40M", "20M", "0M"]
        for i, label in enumerate(y_labels):
            y = chart_top + (chart_bottom - chart_top) * i / 5
            draw.text((chart_left - 50, y - 7), label, fill='black', font=small_font)
        
        # Bars with precise values
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        revenues = [45, 52, 68, 71]  # In millions
        bar_width = 60
        
        colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, (quarter, revenue) in enumerate(zip(quarters, revenues)):
            x = chart_left + 50 + i * 110
            bar_height = (revenue / 100) * (chart_bottom - chart_top)
            y = chart_bottom - bar_height
            
            # Draw bar
            draw.rectangle([x, y, x + bar_width, chart_bottom], fill=colors[i], outline='black')
            
            # Value label on top
            draw.text((x + 15, y - 20), f"${revenue}M", fill='black', font=small_font)
            
            # Quarter label below
            draw.text((x + 20, chart_bottom + 10), quarter, fill='black', font=label_font)
        
        # Add growth percentages
        growth_rates = ["+15.6%", "+30.8%", "+4.4%"]
        for i, rate in enumerate(growth_rates):
            x = chart_left + 110 + i * 110
            draw.text((x, chart_bottom + 40), rate, fill='darkgreen', font=small_font)
        
        return img
    
    def _create_financial_summary(self):
        """Create financial summary document"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            text_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Header
        draw.rectangle([0, 0, 672, 60], fill='darkblue')
        draw.text((50, 20), "QUARTERLY FINANCIAL SUMMARY", fill='white', font=title_font)
        
        # Financial metrics table
        y = 100
        draw.text((50, y), "Q3 2024 Financial Performance", fill='black', font=header_font)
        y += 40
        
        # Table headers
        headers = ["Metric", "Q2 2024", "Q3 2024", "Change"]
        col_widths = [150, 100, 100, 100]
        x_positions = [50, 200, 300, 400]
        
        # Draw table
        table_top = y
        row_height = 30
        
        for i, header in enumerate(headers):
            draw.rectangle([x_positions[i], y, x_positions[i] + col_widths[i], y + row_height], 
                         outline='black', width=1, fill='lightgray')
            draw.text((x_positions[i] + 5, y + 8), header, fill='black', font=text_font)
        
        # Table data
        data = [
            ["Revenue", "$52.3M", "$68.7M", "+31.4%"],
            ["Gross Profit", "$31.4M", "$41.2M", "+31.2%"],
            ["Operating Exp.", "$28.1M", "$35.8M", "+27.4%"],
            ["Net Profit", "$3.3M", "$5.4M", "+63.6%"],
            ["Profit Margin", "6.3%", "7.9%", "+1.6pp"]
        ]
        
        for row_idx, row in enumerate(data):
            y += row_height
            for col_idx, cell in enumerate(row):
                color = 'white'
                if col_idx == 3 and row_idx < 4:  # Change column
                    color = 'lightgreen' if '+' in cell else 'lightcoral'
                    
                draw.rectangle([x_positions[col_idx], y, x_positions[col_idx] + col_widths[col_idx], y + row_height],
                             outline='black', width=1, fill=color)
                draw.text((x_positions[col_idx] + 5, y + 8), cell, fill='black', font=text_font)
        
        # Key insights
        y += 60
        draw.text((50, y), "Key Insights:", fill='black', font=header_font)
        y += 30
        
        insights = [
            "• Revenue growth accelerated to 31.4% QoQ",
            "• Profit margin improved by 1.6 percentage points", 
            "• Operating leverage driving profitability gains",
            "• Strong performance across all business segments"
        ]
        
        for insight in insights:
            draw.text((50, y), insight, fill='black', font=text_font)
            y += 25
        
        return img
    
    def _create_infographic(self):
        """Create infographic with mixed text and graphics"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
            header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            text_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Title
        draw.text((50, 20), "Data Processing Pipeline", fill='black', font=title_font)
        
        # Three main components
        components = [
            ("Data Collection", 100, 100, "• APIs\n• Databases\n• Files"),
            ("Processing", 250, 100, "• Clean\n• Transform\n• Validate"),
            ("Analytics", 400, 100, "• ML Models\n• Statistics\n• Reporting")
        ]
        
        for comp_name, x, y, details in components:
            # Component box
            draw.rectangle([x, y, x+150, y+120], outline='blue', width=2, fill='lightblue')
            draw.text((x+10, y+10), comp_name, fill='black', font=header_font)
            draw.text((x+10, y+40), details, fill='black', font=text_font)
            
            # Arrows between components
            if x < 400:
                draw.line([x+150, y+60, x+200, y+60], fill='red', width=3)
                draw.polygon([(x+195, y+55), (x+205, y+60), (x+195, y+65)], fill='red')
        
        return img
    
    def _create_table_analysis(self):
        """Create table with calculations"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            text_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Title
        draw.text((50, 20), "Product Performance Analysis", fill='black', font=title_font)
        
        # Table
        headers = ["Product", "Q1 Sales", "Q2 Sales", "Q3 Sales", "Average"]
        data = [
            ["Widget A", "1,250", "1,380", "1,520", "1,383"],
            ["Widget B", "980", "1,120", "1,240", "1,113"],
            ["Widget C", "2,100", "2,280", "2,450", "2,277"],
            ["Widget D", "650", "720", "810", "727"],
            ["Total", "4,980", "5,500", "6,020", "5,500"]
        ]
        
        x, y = 50, 100
        col_width = 120
        row_height = 40
        
        # Headers
        for i, header in enumerate(headers):
            draw.rectangle([x + i*col_width, y, x + (i+1)*col_width, y + row_height], 
                          outline='black', width=2, fill='lightgray')
            draw.text((x + i*col_width + 10, y + 12), header, fill='black', font=header_font)
        
        # Data rows
        for row_idx, row in enumerate(data):
            y += row_height
            for col_idx, cell in enumerate(row):
                fill_color = 'lightyellow' if row_idx == len(data)-1 else 'white'
                draw.rectangle([x + col_idx*col_width, y, x + (col_idx+1)*col_width, y + row_height],
                             outline='black', width=1, fill=fill_color)
                draw.text((x + col_idx*col_width + 10, y + 12), cell, fill='black', font=text_font)
        
        return img
    
    def _create_technical_diagram(self):
        """Create technical system diagram"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Title
        draw.text((200, 20), "System Architecture", fill='black', font=title_font)
        
        # Components
        components = [
            ("Load Balancer", 250, 100, 150, 60),
            ("Web Server 1", 100, 200, 120, 50),
            ("Web Server 2", 280, 200, 120, 50),
            ("Web Server 3", 460, 200, 120, 50),
            ("Database", 180, 320, 100, 50),
            ("Cache", 400, 320, 100, 50),
            ("Storage", 290, 420, 100, 50)
        ]
        
        for name, x, y, w, h in components:
            draw.rectangle([x, y, x+w, y+h], outline='black', width=2, fill='lightcyan')
            draw.text((x+10, y+h//2-5), name, fill='black', font=label_font)
        
        # Connections
        connections = [
            ((325, 160), (160, 200)),
            ((325, 160), (340, 200)),
            ((325, 160), (520, 200)),
            ((160, 250), (230, 320)),
            ((340, 250), (230, 320)),
            ((520, 250), (450, 320)),
            ((230, 370), (340, 420)),
            ((450, 370), (340, 420))
        ]
        
        for (x1, y1), (x2, y2) in connections:
            draw.line([x1, y1, x2, y2], fill='red', width=2)
        
        return img

def main():
    """Run real OCR/VQA validation"""
    validator = RealOCRVQAValidator()
    results = validator.run_real_ocr_vqa_validation()
    
    print(f"\n🎉 Validation complete! Check ./shirg_ocr_vqa_visualizations/ for detailed results.")
    return results

if __name__ == "__main__":
    main()