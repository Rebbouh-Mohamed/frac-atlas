#!/usr/bin/env python3
"""
Evaluate Models Without Existing Metrics
Run validation on Original Paper Model and Your Reproduction
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

try:
    from ultralytics import YOLO
    print("✓ Ultralytics imported successfully")
except ImportError:
    print("❌ ERROR: ultralytics not installed")
    print("   Install with: pip install ultralytics")
    sys.exit(1)

# Define paths
BASE_DIR = "/home/rebbouh/data_segm"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")
RAW_OUTPUTS_DIR = os.path.join(ANALYSIS_DIR, "raw_outputs")
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")

# Models to evaluate (these don't have existing metrics)
MODELS_TO_EVALUATE = {
    'Model_1_Original_Paper': {
        'path': os.path.join(BASE_DIR, 'models', 'yolov8_segmentation_fractureAtlas.pt'),
        'description': 'Original published model from FracAtlas paper',
        'architecture': 'YOLOv8 (assumed)',
        'source': 'Paper authors'
    },
    'Model_2_Your_Reproduction': {
        'path': os.path.join(BASE_DIR, 'reproduce', 'runs', 'segment', 'train', 'weights', 'best.pt'),
        'description': 'Your reproduction attempt of their methodology',
        'architecture': 'YOLOv8/YOLO11 (your training)',
        'source': 'Your reproduction'
    }
}

def calculate_comprehensive_metrics(model, data_yaml, model_name):
    """
    Calculate detailed metrics for a model using validation
    
    Args:
        model: YOLO model object
        data_yaml: Path to data.yaml configuration
        model_name: Name identifier for the model
        
    Returns:
        dict: Comprehensive metrics dictionary
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Run validation with conf=0.2 (same as your v7/v8 models)
        print("\nRunning validation with conf=0.2...")
        print("This may take several minutes depending on dataset size...")
        
        val_results = model.val(
            data=data_yaml, 
            split='val', 
            conf=0.2,
            iou=0.7,
            verbose=True,
            save_json=False,
            save_hybrid=False
        )
        
        # Extract metrics from results
        results_dict = val_results.results_dict
        
        print("\n" + "-"*80)
        print("Raw results_dict keys:")
        print(list(results_dict.keys()))
        print("-"*80)
        
        # Build comprehensive metrics dictionary
        metrics = {
            'Model': model_name,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Box Detection Metrics
            'Precision(Box)': float(results_dict.get('metrics/precision(B)', 0)),
            'Recall(Box)': float(results_dict.get('metrics/recall(B)', 0)),
            'mAP50(Box)': float(results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95(Box)': float(results_dict.get('metrics/mAP50-95(B)', 0)),
            
            # Mask Segmentation Metrics
            'Precision(Mask)': float(results_dict.get('metrics/precision(M)', 0)),
            'Recall(Mask)': float(results_dict.get('metrics/recall(M)', 0)),
            'mAP50(Mask)': float(results_dict.get('metrics/mAP50(M)', 0)),
            'mAP50-95(Mask)': float(results_dict.get('metrics/mAP50-95(M)', 0)),
            
            # Performance Metrics
            'Fitness': float(val_results.fitness),
        }
        
        # Add speed metrics if available
        if hasattr(val_results, 'speed') and val_results.speed:
            metrics['Speed_Preprocess(ms)'] = float(val_results.speed.get('preprocess', 0))
            metrics['Speed_Inference(ms)'] = float(val_results.speed.get('inference', 0))
            metrics['Speed_Postprocess(ms)'] = float(val_results.speed.get('postprocess', 0))
            metrics['Total_Speed(ms)'] = (metrics['Speed_Preprocess(ms)'] + 
                                         metrics['Speed_Inference(ms)'] + 
                                         metrics['Speed_Postprocess(ms)'])
        
        # Calculate derived metrics
        p_m = metrics['Precision(Mask)']
        r_m = metrics['Recall(Mask)']
        p_b = metrics['Precision(Box)']
        r_b = metrics['Recall(Box)']
        
        # Mask-based derived metrics
        metrics['F1-Score(Mask)'] = (2 * p_m * r_m) / (p_m + r_m) if (p_m + r_m) > 0 else 0
        metrics['Dice_Coefficient'] = metrics['F1-Score(Mask)']
        metrics['Mean_IoU'] = metrics['mAP50(Mask)']
        
        # Box-based derived metrics
        metrics['F1-Score(Box)'] = (2 * p_b * r_b) / (p_b + r_b) if (p_b + r_b) > 0 else 0
        
        # Overall accuracy approximation
        metrics['Overall_Accuracy'] = (p_m + r_m) / 2
        
        print("\n" + "="*80)
        print("METRICS SUMMARY")
        print("="*80)
        
        key_metrics = [
            'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 'mAP50-95(Mask)',
            'F1-Score(Mask)', 'Dice_Coefficient',
            'Precision(Box)', 'Recall(Box)', 'mAP50(Box)', 'mAP50-95(Box)'
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                print(f"{metric:30s}: {metrics[metric]:.4f}")
        
        print("="*80)
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ ERROR evaluating {model_name}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*80)
    print("FRACATLAS REPRODUCIBILITY STUDY - MODEL EVALUATION")
    print("Evaluating Models 1 & 2 (Original Paper + Your Reproduction)")
    print("="*80)
    
    # Check if data.yaml exists
    if not os.path.exists(DATA_YAML):
        print(f"\n❌ ERROR: data.yaml not found at {DATA_YAML}")
        print("   Please ensure the dataset is properly configured.")
        sys.exit(1)
    
    print(f"\n✓ Found data.yaml: {DATA_YAML}")
    
    # Collect metrics
    all_metrics = []
    
    for model_name, info in MODELS_TO_EVALUATE.items():
        print(f"\n{'#'*80}")
        print(f"MODEL: {model_name}")
        print(f"{'#'*80}")
        print(f"Path: {info['path']}")
        print(f"Description: {info['description']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Source: {info['source']}")
        
        # Check if model exists
        if not os.path.exists(info['path']):
            print(f"\n⚠️  WARNING: Model file not found!")
            print(f"   Expected: {info['path']}")
            print(f"   Skipping this model...")
            continue
        
        try:
            # Load model
            print(f"\nLoading model...")
            model = YOLO(info['path'])
            print(f"✓ Model loaded successfully")
            
            # Evaluate
            metrics = calculate_comprehensive_metrics(model, DATA_YAML, model_name)
            
            if metrics:
                all_metrics.append(metrics)
                
                # Save individual model results
                individual_path = os.path.join(RAW_OUTPUTS_DIR, f"{model_name}_metrics.json")
                with open(individual_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✓ Saved individual metrics to: {individual_path}")
            
        except Exception as e:
            print(f"\n❌ FAILED to load/evaluate {model_name}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    if all_metrics:
        print("\n" + "="*80)
        print("SAVING EVALUATION RESULTS")
        print("="*80)
        
        df_metrics = pd.DataFrame(all_metrics)
        
        # Save as CSV
        csv_path = os.path.join(RESULTS_DIR, 'models_1_2_evaluation_results.csv')
        df_metrics.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(RESULTS_DIR, 'models_1_2_evaluation_results.json')
        df_metrics.to_json(json_path, orient='records', indent=2)
        print(f"✓ Saved JSON to: {json_path}")
        
        # Print comparison table
        print("\n" + "="*80)
        print("EVALUATION RESULTS - MODELS 1 & 2")
        print("="*80)
        
        display_cols = ['Model', 'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 
                       'mAP50-95(Mask)', 'F1-Score(Mask)', 'Dice_Coefficient']
        
        if all(col in df_metrics.columns for col in display_cols):
            print(df_metrics[display_cols].to_string(index=False))
        else:
            print(df_metrics.to_string(index=False))
        
        print("="*80)
        print(f"\n✓ Successfully evaluated {len(all_metrics)} model(s)")
        
    else:
        print("\n" + "="*80)
        print("❌ NO MODELS EVALUATED")
        print("="*80)
        print("\nPossible reasons:")
        print("  1. Model files not found at expected paths")
        print("  2. Ultralytics not installed properly")
        print("  3. data.yaml configuration issues")
        print("\nPlease check the error messages above.")

if __name__ == "__main__":
    main()