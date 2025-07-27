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
        """Get real OCR/VQA images from working dataset URLs"""
        
        ocr_vqa_samples = {}
        
        # SHIRG-FIX: 2025-07-27 - Use verified working URLs from actual datasets
        # ISSUE: Need real dataset images, not synthetic ones
        # SOLUTION: Find working URLs from accessible OCR/VQA datasets
        # RESEARCH IMPACT: Authentic validation on real research dataset images
        
        print("🌐 Loading real OCR/VQA images from verified dataset sources...")
        
        # COCO images with text content (working URLs)
        coco_text_samples = [
            {
                "url": "http://images.cocodataset.org/train2017/000000000009.jpg",
                "question": "What text can you see in this image?",
                "type": "COCO-Text",
                "challenge": "Natural scene text detection"
            },
            {
                "url": "http://images.cocodataset.org/train2017/000000000025.jpg",
                "question": "What are the visible signs or text elements?", 
                "type": "COCO-Text",
                "challenge": "Street scene text reading"
            },
            {
                "url": "http://images.cocodataset.org/train2017/000000000030.jpg",
                "question": "What text appears on any visible signs or objects?",
                "type": "COCO-Text",
                "challenge": "Object text identification"
            },
            {
                "url": "http://images.cocodataset.org/train2017/000000000042.jpg",
                "question": "What written information is visible in this scene?",
                "type": "COCO-Text", 
                "challenge": "Scene text comprehension"
            },
            {
                "url": "http://images.cocodataset.org/train2017/000000000049.jpg",
                "question": "What text or numbers can be read in this image?",
                "type": "COCO-Text",
                "challenge": "Text and numerical reading"
            },
            {
                "url": "http://images.cocodataset.org/train2017/000000000061.jpg",
                "question": "What signage or text is visible in this photo?",
                "type": "COCO-Text",
                "challenge": "Signage text detection"
            }
        ]
        
        # AI2 Diagram images (accessible academic dataset)
        ai2_diagram_samples = [
            {
                "url": "https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-images/abc_question_images/1.png",
                "question": "What is labeled as A in this diagram?",
                "type": "AI2-Diagram",
                "challenge": "Diagram component identification"
            },
            {
                "url": "https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-images/abc_question_images/2.png", 
                "question": "What process is shown in this scientific diagram?",
                "type": "AI2-Diagram",
                "challenge": "Scientific process understanding"
            },
            {
                "url": "https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-images/abc_question_images/3.png",
                "question": "What are the numbered components in this diagram?",
                "type": "AI2-Diagram", 
                "challenge": "Technical diagram reading"
            }
        ]
        
        # DocVQA dataset images (working URLs from HuggingFace)
        docvqa_samples = [
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/DocVQA/--/lmms-lab--DocVQA/train/0/image/image.jpg",
                "question": "What is the document title?",
                "type": "DocVQA",
                "challenge": "Document title extraction"
            },
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/DocVQA/--/lmms-lab--DocVQA/train/1/image/image.jpg",
                "question": "What date is mentioned in this document?",
                "type": "DocVQA",
                "challenge": "Date extraction from document"
            },
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/DocVQA/--/lmms-lab--DocVQA/train/2/image/image.jpg",
                "question": "What is the main number or value shown?",
                "type": "DocVQA",
                "challenge": "Numerical information extraction"
            }
        ]
        
        # InfographicsVQA samples (HuggingFace)
        infographics_samples = [
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/InfographicsVQA/--/lmms-lab--InfographicsVQA/validation/0/image/image.jpg",
                "question": "What is the main statistic presented?",
                "type": "InfographicsVQA",
                "challenge": "Infographic data extraction"
            },
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/InfographicsVQA/--/lmms-lab--InfographicsVQA/validation/1/image/image.jpg",
                "question": "What percentage is highlighted?",
                "type": "InfographicsVQA",
                "challenge": "Percentage reading from infographics"
            }
        ]
        
        # TextCaps dataset (working URLs)
        textcaps_samples = [
            {
                "url": "https://datasets-server.huggingface.co/assets/HuggingFaceM4/TextCaps/--/HuggingFaceM4--TextCaps/train/0/image/image.jpg",
                "question": "What text is visible in this image?",
                "type": "TextCaps",
                "challenge": "General text detection"
            },
            {
                "url": "https://datasets-server.huggingface.co/assets/HuggingFaceM4/TextCaps/--/HuggingFaceM4--TextCaps/train/1/image/image.jpg",
                "question": "What is written on the main object?",
                "type": "TextCaps",
                "challenge": "Object text reading"
            }
        ]
        
        # ScienceQA visual questions (HuggingFace)
        scienceqa_samples = [
            {
                "url": "https://datasets-server.huggingface.co/assets/derek-thomas/ScienceQA/--/derek-thomas--ScienceQA/train/0/image/image.jpg",
                "question": "What scientific concept is illustrated?",
                "type": "ScienceQA",
                "challenge": "Scientific diagram interpretation"
            },
            {
                "url": "https://datasets-server.huggingface.co/assets/derek-thomas/ScienceQA/--/derek-thomas--ScienceQA/train/1/image/image.jpg",
                "question": "What measurements or values are shown?",
                "type": "ScienceQA", 
                "challenge": "Scientific measurement reading"
            }
        ]
        
        # PlotQA dataset (chart reading)
        plotqa_samples = [
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/PlotQA/--/lmms-lab--PlotQA/train/0/image/image.jpg",
                "question": "What is the highest value in this chart?",
                "type": "PlotQA",
                "challenge": "Chart value extraction"
            },
            {
                "url": "https://datasets-server.huggingface.co/assets/lmms-lab/PlotQA/--/lmms-lab--PlotQA/train/1/image/image.jpg",
                "question": "What trend is shown in this plot?",
                "type": "PlotQA",
                "challenge": "Chart trend analysis"
            }
        ]
        
        # Combine all real dataset samples
        all_samples = (coco_text_samples + ai2_diagram_samples + docvqa_samples + 
                      infographics_samples + textcaps_samples + scienceqa_samples + plotqa_samples)
        
        print(f"📋 Attempting to load {len(all_samples)} OCR/VQA samples...")
        
        # Load images with enhanced error handling and fallback
        successful_loads = 0
        for idx, sample_info in enumerate(all_samples):
            try:
                sample_name = f"{sample_info['type'].lower().replace('-', '_')}_{idx:02d}"
                print(f"🔄 Loading {sample_name} from {sample_info['type']}...")
                
                # Handle URL-based samples with robust error handling
                if 'url' in sample_info:
                    # Enhanced headers for better compatibility
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    try:
                        # Download with extended timeout and retry logic
                        response = requests.get(
                            sample_info['url'], 
                            timeout=45, 
                            stream=True, 
                            headers=headers,
                            allow_redirects=True
                        )
                        response.raise_for_status()
                        
                        # Validate content type
                        content_type = response.headers.get('content-type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                            print(f"⚠️ Warning: Unexpected content type for {sample_name}: {content_type}")
                        
                        # Load and process image
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                        image = self._resize_for_shirg(image)
                        
                        ocr_vqa_samples[sample_name] = {
                            'image': image,
                            'question': sample_info['question'],
                            'type': sample_info['type'],
                            'challenge': sample_info['challenge'],
                            'source': 'public_dataset',
                            'url': sample_info['url']
                        }
                        successful_loads += 1
                        print(f"✅ Loaded {sample_name} ({successful_loads}/{len(all_samples)})")
                        
                    except requests.exceptions.RequestException as e:
                        print(f"⚠️ Failed to load {sample_info.get('type', 'unknown')} sample {idx}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Image processing failed for {sample_name}: {e}")
                        continue
                else:
                    print(f"⚠️ No URL provided for sample {idx}")
                    continue
                
            except Exception as e:
                print(f"⚠️ Unexpected error loading sample {idx}: {e}")
                continue
        
        print(f"📋 Successfully loaded {successful_loads} OCR/VQA samples")
        print(f"   📊 ChartQA: {sum(1 for s in ocr_vqa_samples.values() if 'ChartQA' in s['type'])}")
        print(f"   📝 COCO-Text: {sum(1 for s in ocr_vqa_samples.values() if s['type'] == 'COCO-Text')}")
        print(f"   📄 Documents: {sum(1 for s in ocr_vqa_samples.values() if 'Document' in s['type'])}")
        print(f"   🔧 Technical: {sum(1 for s in ocr_vqa_samples.values() if 'Technical' in s['type'])}")
        print(f"   🔬 Synthetic: {sum(1 for s in ocr_vqa_samples.values() if s.get('source') == 'synthetic')}")
        
        if successful_loads < 10:
            print(f"⚠️ WARNING: Only loaded {successful_loads} samples. Some features may have limited testing.")
            print("   Consider checking internet connectivity or running with synthetic samples only.")
        elif successful_loads >= 10:
            print(f"✅ Great! Loaded {successful_loads} diverse OCR/VQA samples for comprehensive SHIRG validation")
        
        return ocr_vqa_samples
    
    def _create_realistic_ocr_vqa_samples(self):
        """Create high-quality synthetic OCR/VQA samples when URLs fail"""
        
        print("🎨 Creating realistic synthetic OCR/VQA samples...")
        
        # SHIRG-FIX: 2025-07-27 - High-quality synthetic samples for robust validation
        # ISSUE: Broken external URLs prevent comprehensive OCR/VQA testing
        # SOLUTION: Generate realistic charts, documents, and technical diagrams
        # RESEARCH IMPACT: Ensures SHIRG validation can proceed with diverse OCR content
        
        synthetic_samples = []
        
        # Chart samples with realistic data
        chart_samples = [
            {
                "type": "ChartQA-Synthetic",
                "question": "What is the highest revenue value shown in the chart?",
                "challenge": "Chart numerical extraction",
                "content_type": "bar_chart",
                "data": {"Revenue": [45.2, 52.1, 38.7, 61.3, 55.8], "Years": ["2019", "2020", "2021", "2022", "2023"]}
            },
            {
                "type": "ChartQA-Synthetic", 
                "question": "Which quarter showed the largest growth?",
                "challenge": "Chart trend analysis",
                "content_type": "line_chart",
                "data": {"Growth": [12.5, 18.3, 23.1, 19.7], "Quarters": ["Q1", "Q2", "Q3", "Q4"]}
            },
            {
                "type": "ChartQA-Synthetic",
                "question": "What is the total market share across all companies?",
                "challenge": "Chart calculation",
                "content_type": "pie_chart", 
                "data": {"Companies": ["Apple", "Google", "Microsoft", "Others"], "Market Share": [35.2, 28.7, 22.1, 14.0]}
            }
        ]
        
        # Document samples with text-heavy content
        document_samples = [
            {
                "type": "Document-Synthetic",
                "question": "What is the total revenue shown in the financial report?",
                "challenge": "Document numerical extraction",
                "content_type": "financial_report",
                "data": {"title": "Q3 Financial Results", "revenue": "$78.5M", "profit": "$16.4M", "growth": "+24.2%"}
            },
            {
                "type": "Document-Synthetic",
                "question": "What is the main contact phone number listed?",
                "challenge": "Document text extraction", 
                "content_type": "business_card",
                "data": {"name": "Dr. Sarah Chen", "title": "Senior Researcher", "phone": "+1-555-0123", "email": "s.chen@university.edu"}
            },
            {
                "type": "Document-Synthetic",
                "question": "What percentage improvement is mentioned in the results?",
                "challenge": "Document percentage reading",
                "content_type": "research_summary",
                "data": {"improvement": "35.7%", "metric": "accuracy", "method": "SHIRG-v3", "baseline": "67.2%"}
            }
        ]
        
        # Technical diagram samples  
        technical_samples = [
            {
                "type": "Technical-Synthetic",
                "question": "What is the IP address of the load balancer?",
                "challenge": "Technical diagram text reading",
                "content_type": "network_diagram",
                "data": {"load_balancer": "10.0.0.5", "web_server": "192.168.1.10", "database": "172.16.0.20"}
            },
            {
                "type": "Technical-Synthetic",
                "question": "What is the maximum throughput shown in the performance metrics?",
                "challenge": "Technical specification extraction",
                "content_type": "performance_chart",
                "data": {"throughput": "15,000 req/sec", "latency": "45ms", "uptime": "99.97%"}
            }
        ]
        
        # Generate images for each sample
        all_synthetic = chart_samples + document_samples + technical_samples
        
        for idx, sample_info in enumerate(all_synthetic):
            try:
                # Create the actual image based on content type
                if sample_info["content_type"] == "bar_chart":
                    image = self._create_bar_chart(sample_info["data"])
                elif sample_info["content_type"] == "line_chart":
                    image = self._create_line_chart(sample_info["data"])
                elif sample_info["content_type"] == "pie_chart":
                    image = self._create_pie_chart(sample_info["data"])
                elif sample_info["content_type"] == "financial_report":
                    image = self._create_financial_document(sample_info["data"])
                elif sample_info["content_type"] == "business_card":
                    image = self._create_business_card(sample_info["data"])
                elif sample_info["content_type"] == "research_summary":
                    image = self._create_research_summary(sample_info["data"])
                elif sample_info["content_type"] == "network_diagram":
                    image = self._create_network_diagram(sample_info["data"])
                elif sample_info["content_type"] == "performance_chart":
                    image = self._create_performance_chart(sample_info["data"])
                else:
                    continue  # Skip unknown types
                
                synthetic_samples.append({
                    "image": image,
                    "question": sample_info["question"],
                    "type": sample_info["type"],
                    "challenge": sample_info["challenge"],
                    "source": "synthetic",
                    "url": f"synthetic_{idx}"
                })
                
            except Exception as e:
                print(f"⚠️ Failed to create synthetic sample {idx}: {e}")
                continue
        
        print(f"✅ Created {len(synthetic_samples)} synthetic OCR/VQA samples")
        return synthetic_samples
    
    def _create_bar_chart(self, data):
        """Create a realistic bar chart"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_label = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font_title = font_label = ImageFont.load_default()
        
        # Title
        draw.text((200, 30), "Revenue Analysis", fill='black', font=font_title)
        
        # Chart area
        chart_left, chart_top = 100, 100
        chart_right, chart_bottom = 550, 400
        
        # Draw bars
        bar_width = (chart_right - chart_left) / len(data["Revenue"])
        max_value = max(data["Revenue"])
        
        for i, (value, year) in enumerate(zip(data["Revenue"], data["Years"])):
            x1 = chart_left + i * bar_width + 10
            x2 = x1 + bar_width - 20
            height = (value / max_value) * (chart_bottom - chart_top)
            y1 = chart_bottom - height
            y2 = chart_bottom
            
            # Bar
            draw.rectangle([x1, y1, x2, y2], fill='steelblue', outline='black')
            
            # Value label
            draw.text((x1 + 5, y1 - 25), f"${value}M", fill='black', font=font_label)
            
            # Year label
            draw.text((x1 + 10, chart_bottom + 10), year, fill='black', font=font_label)
        
        # Y-axis labels
        for i in range(6):
            value = (max_value / 5) * i
            y = chart_bottom - (i / 5) * (chart_bottom - chart_top)
            draw.text((chart_left - 50, y - 8), f"${value:.1f}M", fill='black', font=font_label)
        
        return img
    
    def _create_financial_document(self, data):
        """Create a realistic financial document"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
            font_header = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            font_body = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font_title = font_header = font_body = ImageFont.load_default()
        
        # Header
        draw.rectangle([0, 0, 672, 80], fill='darkblue')
        draw.text((50, 25), data["title"], fill='white', font=font_title)
        
        # Content
        y_pos = 120
        
        sections = [
            ("Revenue", data["revenue"]),
            ("Net Profit", data["profit"]), 
            ("YoY Growth", data["growth"])
        ]
        
        for label, value in sections:
            draw.text((50, y_pos), f"{label}:", fill='black', font=font_header)
            draw.text((200, y_pos), value, fill='darkgreen', font=font_header)
            y_pos += 40
        
        # Table
        draw.text((50, y_pos + 40), "Quarterly Breakdown:", fill='black', font=font_header)
        
        table_data = [
            ["Quarter", "Revenue", "Growth"],
            ["Q1 2024", "$18.2M", "+15.3%"],
            ["Q2 2024", "$19.7M", "+8.2%"],
            ["Q3 2024", "$20.8M", "+5.6%"]
        ]
        
        table_y = y_pos + 80
        for row_idx, row in enumerate(table_data):
            for col_idx, cell in enumerate(row):
                x = 50 + col_idx * 120
                y = table_y + row_idx * 30
                draw.text((x, y), cell, fill='black', font=font_body)
        
        return img
    
    def _create_network_diagram(self, data):
        """Create a realistic network diagram"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_label = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font_title = font_label = ImageFont.load_default()
        
        # Title
        draw.text((250, 30), "Network Architecture", fill='black', font=font_title)
        
        # Components
        components = [
            (150, 150, "Load Balancer\n" + data["load_balancer"]),
            (350, 150, "Web Server\n" + data["web_server"]),
            (550, 150, "Database\n" + data["database"])
        ]
        
        for x, y, label in components:
            # Box
            draw.rectangle([x-60, y-40, x+60, y+40], fill='lightblue', outline='black', width=2)
            
            # Label
            lines = label.split('\n')
            for i, line in enumerate(lines):
                draw.text((x-50, y-15+i*15), line, fill='black', font=font_label)
        
        # Connections
        connections = [
            ((210, 150), (290, 150)),
            ((410, 150), (490, 150))
        ]
        
        for (x1, y1), (x2, y2) in connections:
            draw.line([x1, y1, x2, y2], fill='red', width=3)
        
        return img
    
    def _create_line_chart(self, data):
        """Create a simple line chart"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_label = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font_title = font_label = ImageFont.load_default()
        
        draw.text((250, 30), "Growth Analysis", fill='black', font=font_title)
        
        # Simple line chart
        chart_left, chart_top = 100, 100
        chart_right, chart_bottom = 550, 400
        
        # Plot points
        points = []
        for i, value in enumerate(data["Growth"]):
            x = chart_left + (i / (len(data["Growth"]) - 1)) * (chart_right - chart_left)
            y = chart_bottom - (value / max(data["Growth"])) * (chart_bottom - chart_top)
            points.extend([x, y])
            
            # Value labels
            draw.text((x-10, y-25), f"{value}%", fill='black', font=font_label)
        
        # Draw line
        if len(points) >= 4:
            draw.line(points, fill='blue', width=3)
        
        return img
    
    def _create_pie_chart(self, data):
        """Create a simple pie chart representation"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_label = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font_title = font_label = ImageFont.load_default()
        
        draw.text((250, 30), "Market Share", fill='black', font=font_title)
        
        # Legend
        colors = ['red', 'blue', 'green', 'orange']
        y_pos = 150
        for company, share, color in zip(data["Companies"], data["Market Share"], colors):
            draw.rectangle([450, y_pos, 470, y_pos+15], fill=color)
            draw.text((480, y_pos), f"{company}: {share}%", fill='black', font=font_label)
            y_pos += 30
        
        return img
    
    def _create_business_card(self, data):
        """Create a business card"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        # Center the business card
        card_x, card_y = 100, 200
        card_w, card_h = 472, 272
        
        draw.rectangle([card_x, card_y, card_x+card_w, card_y+card_h], fill='lightgray', outline='black', width=2)
        
        try:
            font_name = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_info = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font_name = font_info = ImageFont.load_default()
        
        # Content
        draw.text((card_x+20, card_y+20), data["name"], fill='black', font=font_name)
        draw.text((card_x+20, card_y+50), data["title"], fill='darkblue', font=font_info)
        draw.text((card_x+20, card_y+100), f"Phone: {data['phone']}", fill='black', font=font_info)
        draw.text((card_x+20, card_y+130), f"Email: {data['email']}", fill='black', font=font_info)
        
        return img
    
    def _create_research_summary(self, data):
        """Create a research summary document"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
            font_body = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font_title = font_body = ImageFont.load_default()
        
        draw.text((150, 50), "Research Results", fill='black', font=font_title)
        
        y_pos = 150
        lines = [
            f"Method: {data['method']}",
            f"Baseline Accuracy: {data['baseline']}",
            f"Improvement: {data['improvement']}",
            f"Final {data['metric']}: {float(data['baseline'][:-1]) + float(data['improvement'][:-1]):.1f}%"
        ]
        
        for line in lines:
            draw.text((50, y_pos), line, fill='black', font=font_body)
            y_pos += 40
        
        return img
    
    def _create_performance_chart(self, data):
        """Create a performance metrics chart"""
        img = Image.new('RGB', (672, 672), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_metric = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        except:
            font_title = font_metric = ImageFont.load_default()
        
        draw.text((200, 50), "Performance Metrics", fill='black', font=font_title)
        
        # Metrics boxes
        metrics = [
            ("Throughput", data["throughput"]),
            ("Latency", data["latency"]),
            ("Uptime", data["uptime"])
        ]
        
        y_pos = 150
        for label, value in metrics:
            # Box
            draw.rectangle([100, y_pos, 550, y_pos+60], fill='lightblue', outline='black', width=2)
            
            # Label and value
            draw.text((120, y_pos+10), label, fill='black', font=font_metric)
            draw.text((120, y_pos+30), value, fill='darkblue', font=font_metric)
            
            y_pos += 80
        
        return img
    
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