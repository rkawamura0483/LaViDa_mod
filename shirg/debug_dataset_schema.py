#!/usr/bin/env python3
"""
Debug script to inspect dataset schemas and field names
"""

from datasets import load_dataset
import json

def inspect_dataset(dataset_name, config=None, split="train", max_items=5):
    """Inspect a dataset and print its schema"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {dataset_name} (config={config}, split={split})")
    print(f"{'='*60}")
    
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # Get first few items
        items = []
        for i, item in enumerate(dataset):
            if i >= max_items:
                break
            items.append(item)
        
        if items:
            # Print available keys
            print(f"\nAvailable keys: {list(items[0].keys())}")
            
            # Print sample structure
            print(f"\nFirst item structure:")
            for key, value in items[0].items():
                value_type = type(value).__name__
                if isinstance(value, str):
                    value_preview = value[:100] + "..." if len(value) > 100 else value
                elif isinstance(value, list):
                    value_preview = f"List with {len(value)} items"
                elif isinstance(value, dict):
                    value_preview = f"Dict with keys: {list(value.keys())}"
                else:
                    value_preview = str(value)[:100]
                
                print(f"  '{key}': {value_type} = {value_preview}")
            
            # Look for question/answer fields
            print(f"\nQuestion/Answer field analysis:")
            question_fields = [k for k in items[0].keys() if 'question' in k.lower() or 'query' in k.lower()]
            answer_fields = [k for k in items[0].keys() if 'answer' in k.lower() or 'response' in k.lower()]
            
            if question_fields:
                print(f"  Potential question fields: {question_fields}")
            else:
                print(f"  No 'question' field found. All fields: {list(items[0].keys())}")
            
            if answer_fields:
                print(f"  Potential answer fields: {answer_fields}")
            else:
                print(f"  No 'answer' field found")
                
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Main inspection
if __name__ == "__main__":
    print("🔍 Inspecting dataset schemas for SHIRG training...")
    
    # ChartQA
    inspect_dataset("ahmed-masry/ChartQA", split="train")
    
    # DocVQA
    inspect_dataset("lmms-lab/DocVQA", config="DocVQA", split="validation")
    
    # VQA v2 - try different sources
    inspect_dataset("HuggingFaceM4/VQAv2", split="train")
    inspect_dataset("Graphcore/vqa", split="train")
    
    # TextVQA
    inspect_dataset("lmms-lab/textvqa", split="train")
    
    # OCR-VQA
    inspect_dataset("howard-hou/OCR-VQA", split="train")
    
    # InfoVQA
    inspect_dataset("vidore/infovqa_train", split="train")