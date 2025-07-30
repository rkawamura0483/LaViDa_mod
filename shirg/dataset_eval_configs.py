#!/usr/bin/env python3
"""
Dataset Evaluation Configuration Loader
Loads the official lmms-eval prompts and generation parameters for each dataset
"""

import os
import yaml
from typing import Dict, Any, Optional, List

class DatasetEvalConfig:
    """Loads and manages evaluation configurations for different datasets"""
    
    # Map dataset types to their YAML config files and evaluation metrics
    DATASET_CONFIG_MAP = {
        "DocVQA": {
            "yaml": "eval/lmms_eval/tasks/docvqa/_default_template_docvqa_yaml",
            "metrics": ["anls"]  # Average Normalized Levenshtein Similarity
        },
        "ChartQA": {
            "yaml": "eval/lmms_eval/tasks/chartqa/chartqa.yaml",
            "metrics": ["relaxed_accuracy"]  # Relaxed accuracy (±5% for numbers)
        },
        "InfoVQA": {
            "yaml": "eval/lmms_eval/tasks/infovqa/_default_template_infovqa_yaml",
            "metrics": ["anls"]  # Average Normalized Levenshtein Similarity
        },
        "TextVQA": {
            "yaml": "eval/lmms_eval/tasks/textvqa/textvqa_val.yaml",
            "metrics": ["accuracy"]  # VQA accuracy metric
        },
        "VQAv2": {
            "yaml": "eval/lmms_eval/tasks/vqav2/vqav2_val.yaml",
            "metrics": ["vqa_accuracy"]  # VQA v2 accuracy (3/10 human agreement)
        },
        "MathVista": {
            "yaml": "eval/lmms_eval/tasks/mathvista/mathvista_testmini.yaml",
            "metrics": ["accuracy", "gpt_eval"]  # Math accuracy
        },
        "MathVerse": {
            "yaml": "eval/lmms_eval/tasks/mathverse/mathverse_testmini.yaml",
            "metrics": ["accuracy", "gpt_eval"]  # Math accuracy
        },
        "OCR-VQA": {
            "yaml": None,  # No official lmms-eval config
            "metrics": ["accuracy"]  # Standard accuracy
        }
    }
    
    # Default configurations if YAML not found
    DEFAULT_CONFIGS = {
        "default": {
            "pre_prompt": "",
            "post_prompt": "\nAnswer the question using a single word or phrase.",
            "generation_kwargs": {
                "max_new_tokens": 32,
                "temperature": 0,
                "do_sample": False
            }
        },
        "math": {
            "pre_prompt": "",
            "post_prompt": "\nAnswer the question using a number, expression, or short phrase.",
            "generation_kwargs": {
                "max_new_tokens": 64,
                "temperature": 0,
                "do_sample": False
            }
        }
    }
    
    def __init__(self, base_path: str = "./"):
        """Initialize with base path to LaViDa repository"""
        self.base_path = base_path
        self._config_cache = {}
    
    def load_yaml_config(self, yaml_path: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        full_path = os.path.join(self.base_path, yaml_path)
        
        if not os.path.exists(full_path):
            print(f"⚠️ Config file not found: {full_path}")
            return {}
            
        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️ Error loading {full_path}: {e}")
            return {}
    
    def get_dataset_config(self, dataset_type: str) -> Dict[str, Any]:
        """Get evaluation configuration for a specific dataset type"""
        
        # Check cache first
        if dataset_type in self._config_cache:
            return self._config_cache[dataset_type]
        
        # Get config info for dataset
        dataset_info = self.DATASET_CONFIG_MAP.get(dataset_type, {})
        yaml_path = dataset_info.get('yaml') if isinstance(dataset_info, dict) else None
        metrics = dataset_info.get('metrics', ['accuracy']) if isinstance(dataset_info, dict) else ['accuracy']
        
        if yaml_path is None:
            # Use default config for datasets without specific config
            if "math" in dataset_type.lower():
                config = self.DEFAULT_CONFIGS["math"].copy()
            else:
                config = self.DEFAULT_CONFIGS["default"].copy()
            print(f"📋 Using default config for {dataset_type}")
        else:
            # Load from YAML
            yaml_config = self.load_yaml_config(yaml_path)
            
            # Extract relevant fields
            config = {
                "generation_kwargs": yaml_config.get("generation_kwargs", {}),
                "pre_prompt": "",
                "post_prompt": ""
            }
            
            # Get prompts from lmms_eval_specific_kwargs
            if "lmms_eval_specific_kwargs" in yaml_config:
                default_kwargs = yaml_config["lmms_eval_specific_kwargs"].get("default", {})
                config["pre_prompt"] = default_kwargs.get("pre_prompt", "")
                config["post_prompt"] = default_kwargs.get("post_prompt", "")
            
            # Handle includes (for configs that reference other templates)
            if "include" in yaml_config and not config["post_prompt"]:
                include_path = yaml_config["include"]
                if not include_path.endswith('.yaml'):
                    include_path = f"eval/lmms_eval/tasks/{dataset_type.lower()}/{include_path}"
                
                included_config = self.load_yaml_config(include_path)
                if "lmms_eval_specific_kwargs" in included_config:
                    default_kwargs = included_config["lmms_eval_specific_kwargs"].get("default", {})
                    config["pre_prompt"] = default_kwargs.get("pre_prompt", "")
                    config["post_prompt"] = default_kwargs.get("post_prompt", "")
                
                # Also get generation kwargs from included file if not in main
                if not config["generation_kwargs"] and "generation_kwargs" in included_config:
                    config["generation_kwargs"] = included_config["generation_kwargs"]
            
            print(f"📋 Loaded config for {dataset_type} from {yaml_path}")
        
        # Add metrics to config
        config['metrics'] = metrics
        
        # Cache the config
        self._config_cache[dataset_type] = config
        
        # Print loaded configuration
        print(f"   Pre-prompt: '{config['pre_prompt']}'")
        print(f"   Post-prompt: '{config['post_prompt']}'")
        print(f"   Generation: {config['generation_kwargs']}")
        print(f"   Metrics: {config['metrics']}")
        
        return config
    
    def format_question_with_prompts(self, question: str, dataset_type: str) -> str:
        """Format a question with the appropriate pre/post prompts for the dataset"""
        config = self.get_dataset_config(dataset_type)
        return f"{config['pre_prompt']}{question}{config['post_prompt']}"
    
    def get_generation_kwargs(self, dataset_type: str) -> Dict[str, Any]:
        """Get generation kwargs for a specific dataset"""
        config = self.get_dataset_config(dataset_type)
        return config.get("generation_kwargs", {})
    
    def get_metrics(self, dataset_type: str) -> List[str]:
        """Get evaluation metrics for a specific dataset"""
        config = self.get_dataset_config(dataset_type)
        return config.get("metrics", ["accuracy"])


# Test the configuration loader
if __name__ == "__main__":
    loader = DatasetEvalConfig("../")
    
    print("🧪 Testing dataset configuration loader...")
    print("="*60)
    
    test_datasets = ["DocVQA", "ChartQA", "InfoVQA", "TextVQA", "VQAv2", "MathVista", "OCR-VQA"]
    
    for dataset in test_datasets:
        print(f"\n📊 {dataset}:")
        config = loader.get_dataset_config(dataset)
        
        # Test question formatting
        test_question = "What is the total value?"
        formatted = loader.format_question_with_prompts(test_question, dataset)
        print(f"   Formatted: {formatted}")
        print()