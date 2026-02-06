#!/usr/bin/env python3
"""
Load Existing Metrics for Models 3 & 4
Extract comprehensive metrics from v7 and v8 results
"""

import os
import pandas as pd
import json
from datetime import datetime

# Define paths
BASE_DIR = "/home/rebbouh/data_segm"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")
RAW_OUTPUTS_DIR = os.path.join(ANALYSIS_DIR, "raw_outputs")

# Models with existing metrics
MODELS_WITH_METRICS = {
    'Model_3_Your_v7': {
        'comprehensive_metrics': os.path.join(BASE_DIR, 'v7', 'results', 'comprehensive_metrics.csv'),
        'training_results': os.path.join(BASE_DIR, 'v7', 'bone_fracture_model', 'results.csv'),
        'description': 'Your enhanced model with corrections + augmentation',
        'architecture': 'YOLO11m-seg',
        'imgsz': 800,
        'epochs': 100,
        'source': 'Your work'
    },
    'Model_4_Your_v8': {
        'comprehensive_metrics': os.path.join(BASE_DIR, 'v8', 'results', 'comprehensive_metrics.csv'),
        'training_results': os.path.join(BASE_DIR, 'v8', 'bone_fracture_model', 'results.csv'),
        'description': 'Your optimized model with extended training',
        'architecture': 'YOLO11m-seg',
        'imgsz': 640,
        'epochs': 120,
        'source': 'Your work'
    }
}

def load_comprehensive_metrics(csv_path, model_name):
    """Load comprehensive metrics from CSV file"""
    print(f"\n{'─'*80}")
    print(f"Loading: {model_name}")
    print(f"Path: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"⚠️  File not found!")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print(f"⚠️  Empty CSV file")
            return None
        
        # Get first row (should be only row for comprehensive metrics)
        metrics = df.iloc[0].to_dict()
        metrics['Model'] = model_name
        metrics['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"✓ Loaded successfully")
        print(f"  - Metrics available: {len(metrics)}")
        
        # Print key metrics
        key_metrics = [
            'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 
            'mAP50-95(Mask)', 'F1-Score(Mask)', 'Dice_Coefficient'
        ]
        
        print("\n  Key Metrics:")
        for key in key_metrics:
            if key in metrics:
                print(f"    {key:25s}: {metrics[key]:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_training_summary(csv_path, model_name):
    """Load training history from results.csv"""
    print(f"\n  Loading training history...")
    print(f"  Path: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  Training results not found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ Training history loaded ({len(df)} epochs)")
        
        # Get final epoch stats
        final_epoch = df.iloc[-1].to_dict()
        
        summary = {
            'total_epochs': len(df),
            'final_train_loss': final_epoch.get('train/box_loss', 0) + 
                               final_epoch.get('train/seg_loss', 0) + 
                               final_epoch.get('train/cls_loss', 0),
            'final_val_loss': final_epoch.get('val/box_loss', 0) + 
                             final_epoch.get('val/seg_loss', 0) + 
                             final_epoch.get('val/cls_loss', 0),
        }
        
        print(f"    Total epochs trained: {summary['total_epochs']}")
        print(f"    Final training loss: {summary['final_train_loss']:.4f}")
        print(f"    Final validation loss: {summary['final_val_loss']:.4f}")
        
        return summary
        
    except Exception as e:
        print(f"  ❌ Error loading training history: {str(e)}")
        return None

def main():
    print("\n" + "="*80)
    print("FRACATLAS REPRODUCIBILITY STUDY - LOAD EXISTING METRICS")
    print("Loading metrics for Models 3 & 4 (Your v7 and v8)")
    print("="*80)
    
    all_metrics = []
    model_details = []
    
    for model_name, info in MODELS_WITH_METRICS.items():
        print(f"\n{'#'*80}")
        print(f"MODEL: {model_name}")
        print(f"{'#'*80}")
        print(f"Description: {info['description']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Image Size: {info['imgsz']}")
        print(f"Epochs: {info['epochs']}")
        print(f"Source: {info['source']}")
        
        # Store model configuration
        model_details.append({
            'Model': model_name,
            'Architecture': info['architecture'],
            'Image_Size': info['imgsz'],
            'Epochs': info['epochs'],
            'Description': info['description']
        })
        
        # Load comprehensive metrics
        metrics = load_comprehensive_metrics(info['comprehensive_metrics'], model_name)
        
        if metrics:
            # Load training summary
            training_summary = load_training_summary(info['training_results'], model_name)
            
            if training_summary:
                metrics.update({
                    'Total_Epochs_Trained': training_summary['total_epochs'],
                    'Final_Train_Loss': training_summary['final_train_loss'],
                    'Final_Val_Loss': training_summary['final_val_loss']
                })
            
            all_metrics.append(metrics)
            
            # Save individual model results
            individual_path = os.path.join(RAW_OUTPUTS_DIR, f"{model_name}_metrics.json")
            with open(individual_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✓ Saved to: {individual_path}")
    
    # Save results
    if all_metrics:
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save metrics
        df_metrics = pd.DataFrame(all_metrics)
        
        csv_path = os.path.join(RESULTS_DIR, 'models_3_4_existing_metrics.csv')
        df_metrics.to_csv(csv_path, index=False)
        print(f"✓ Saved metrics CSV to: {csv_path}")
        
        json_path = os.path.join(RESULTS_DIR, 'models_3_4_existing_metrics.json')
        df_metrics.to_json(json_path, orient='records', indent=2)
        print(f"✓ Saved metrics JSON to: {json_path}")
        
        # Save model configurations
        df_details = pd.DataFrame(model_details)
        details_path = os.path.join(RESULTS_DIR, 'models_3_4_configurations.csv')
        df_details.to_csv(details_path, index=False)
        print(f"✓ Saved configurations to: {details_path}")
        
        # Print comparison table
        print("\n" + "="*80)
        print("METRICS COMPARISON - MODELS 3 & 4")
        print("="*80)
        
        display_cols = ['Model', 'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 
                       'mAP50-95(Mask)', 'F1-Score(Mask)', 'Dice_Coefficient']
        
        if all(col in df_metrics.columns for col in display_cols):
            print(df_metrics[display_cols].to_string(index=False))
        else:
            # Show first 10 columns
            print(df_metrics.iloc[:, :10].to_string(index=False))
        
        print("="*80)
        
        # Compare v7 vs v8
        if len(all_metrics) == 2:
            print("\n" + "="*80)
            print("IMPROVEMENTS: v8 vs v7")
            print("="*80)
            
            v7_metrics = all_metrics[0]
            v8_metrics = all_metrics[1]
            
            comparison_metrics = [
                'Precision(Mask)', 'Recall(Mask)', 'mAP50(Mask)', 
                'mAP50-95(Mask)', 'F1-Score(Mask)'
            ]
            
            for metric in comparison_metrics:
                if metric in v7_metrics and metric in v8_metrics:
                    v7_val = v7_metrics[metric]
                    v8_val = v8_metrics[metric]
                    diff = v8_val - v7_val
                    pct = (diff / v7_val * 100) if v7_val != 0 else 0
                    symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                    
                    print(f"{metric:25s}: {v7_val:.4f} → {v8_val:.4f}  "
                          f"{symbol} {abs(pct):+.2f}%")
            
            print("="*80)
        
        print(f"\n✓ Successfully loaded metrics for {len(all_metrics)} model(s)")
        
    else:
        print("\n" + "="*80)
        print("❌ NO METRICS LOADED")
        print("="*80)
        print("\nExpected files:")
        for model_name, info in MODELS_WITH_METRICS.items():
            print(f"\n{model_name}:")
            print(f"  - {info['comprehensive_metrics']}")
            print(f"  - {info['training_results']}")
        print("\nPlease check if these files exist.")

if __name__ == "__main__":
    main()