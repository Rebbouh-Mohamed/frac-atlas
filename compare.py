#!/usr/bin/env python3
"""
FIXED MODEL COMPARISON SCRIPT
Handles path issues and provides better diagnostics
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# ============================================================================
# AUTO-DETECTION AND CONFIGURATION
# ============================================================================

def find_data_yaml():
    """Try to find data.yaml in common locations"""
    possible_paths = [
        './data.yaml',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def find_masks_dir(base_dir):
    """Find actual masks directory"""
    possible_paths = [
        os.path.join(base_dir, 'valid/masks'),
        os.path.join(base_dir, 'val/masks'),
        os.path.join(base_dir, 'test/masks'),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

CONFIG = {
    # Model paths
    'unet_model_path': './models/best_model.pth',
    'yolo_model_path': './models/v7_best_mine.pt',
    
    # Dataset paths - will auto-detect if not found
    'data_yaml': None,  # Will auto-detect
    'test_images_dir': './results-unet/data_unet/valid/images',
    'test_masks_dir': "./results-unet/data_unet/valid/masks",  # Will auto-detect
    
    # Output directory
    'output_dir': './outputs/model_comparison1',
    
    # Evaluation settings
    'img_size': 640,
    'conf_threshold': 0.2,
    'iou_threshold': 0.5,
    'device': 'cpu'
}

# ============================================================================
# YOLO EVALUATION WITH FIXED PATHS
# ============================================================================

def create_temp_data_yaml(original_yaml_path, output_dir):
    """Create a corrected data.yaml with proper paths"""
    import yaml
    
    # Read original yaml
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Find actual data directories
    base_paths = [
        '.',
    ]
    
    for base_path in base_paths:
        # Try to find train/val/test directories
        for split in ['train', 'val', 'valid', 'test']:
            img_dir = os.path.join(base_path, split, 'images')
            if os.path.exists(img_dir):
                print(f"  Found {split} images: {img_dir}")
                data[split] = img_dir
    
    # Save corrected yaml
    temp_yaml = os.path.join(output_dir, 'data_corrected.yaml')
    with open(temp_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"  ✓ Created corrected YAML: {temp_yaml}")
    return temp_yaml

def evaluate_yolo_model(model_path, data_yaml, output_dir):
    """Evaluate YOLO with better error handling"""
    print("\n" + "="*80)
    print("🔍 EVALUATING YOLO MODEL")
    print("="*80)
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported")
    except ImportError:
        print("❌ ERROR: Install ultralytics: pip install ultralytics")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Auto-detect data.yaml if not provided
    if data_yaml is None or not os.path.exists(data_yaml):
        print("\n🔍 Auto-detecting data.yaml...")
        data_yaml = find_data_yaml()
        if data_yaml:
            print(f"  ✓ Found: {data_yaml}")
        else:
            print("❌ Could not find data.yaml")
            print("Please create one or specify the correct path")
            return None
    
    print(f"\n📁 Model: {model_path}")
    print(f"📁 Data YAML: {data_yaml}")
    
    try:
        model = YOLO(model_path)
        print("✓ Model loaded")
        
        # Try direct validation first
        print("\n🔄 Running validation...")
        try:
            results = model.val(
                data=data_yaml,
                split='val',
                conf=CONFIG['conf_threshold'],
                iou=CONFIG['iou_threshold'],
                verbose=False
            )
        except FileNotFoundError as e:
            print(f"\n⚠️  Path error detected. Trying to fix...")
            # Try to create corrected yaml
            try:
                import yaml
                corrected_yaml = create_temp_data_yaml(data_yaml, output_dir)
                results = model.val(
                    data=corrected_yaml,
                    split='val',
                    conf=CONFIG['conf_threshold'],
                    iou=CONFIG['iou_threshold'],
                    verbose=False
                )
            except Exception as fix_error:
                print(f"❌ Could not fix paths: {fix_error}")
                raise e
        
        # Extract metrics
        results_dict = results.results_dict
        
        metrics = {
            'Model': 'YOLO Segmentation',
            'Architecture': 'YOLOv8',
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            'Precision(Mask)': float(results_dict.get('metrics/precision(M)', 0)),
            'Recall(Mask)': float(results_dict.get('metrics/recall(M)', 0)),
            'mAP50(Mask)': float(results_dict.get('metrics/mAP50(M)', 0)),
            'mAP50-95(Mask)': float(results_dict.get('metrics/mAP50-95(M)', 0)),
            'Fitness': float(results.fitness),
        }
        
        # Calculate derived metrics
        p = metrics['Precision(Mask)']
        r = metrics['Recall(Mask)']
        metrics['F1-Score(Mask)'] = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        metrics['Dice_Coefficient'] = metrics['F1-Score(Mask)']
        
        # Speed
        if hasattr(results, 'speed') and results.speed:
            metrics['Speed_Inference(ms)'] = float(results.speed.get('inference', 0))
        
        print("\n✅ YOLO EVALUATION COMPLETE")
        print_metrics_summary(metrics)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'yolo_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Saved: {results_path}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ YOLO evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# UNet++ EVALUATION WITH FIXED MASK LOADING
# ============================================================================

def evaluate_unet_model(model_path, test_images_dir, test_masks_dir, output_dir):
    """Evaluate UNet++ with better mask path handling"""
    print("\n" + "="*80)
    print("🔍 EVALUATING UNet++ MODEL")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        import segmentation_models_pytorch as smp
        import cv2
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print("✓ Dependencies imported")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Images dir not found: {test_images_dir}")
        return None
    
    # Auto-detect masks directory
    if test_masks_dir is None or not os.path.exists(test_masks_dir):
        print("\n🔍 Auto-detecting masks directory...")
        base_dir = os.path.dirname(test_images_dir)
        test_masks_dir = find_masks_dir(base_dir)
        if test_masks_dir:
            print(f"  ✓ Found: {test_masks_dir}")
        else:
            print(f"❌ Could not find masks directory near {test_images_dir}")
            return None
    
    print(f"\n📁 Model: {model_path}")
    print(f"📁 Images: {test_images_dir}")
    print(f"📁 Masks: {test_masks_dir}")
    
    try:
        device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {device}")
        
        # Load model
        print("\nLoading UNet++ model...")
        model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-b5',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        print("✓ Model loaded")
        
        # FIXED Dataset class
        class TestDataset(Dataset):
            def __init__(self, images_dir, masks_dir, img_size=640):
                self.images_dir = Path(images_dir)
                self.masks_dir = Path(masks_dir)
                self.img_size = img_size
                
                # Get image files
                self.image_files = sorted(
                    list(self.images_dir.glob('*.jpg')) + 
                    list(self.images_dir.glob('*.png'))
                )
                
                # Verify we have images
                if len(self.image_files) == 0:
                    raise ValueError(f"No images found in {images_dir}")
                
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                # Load image
                img_path = self.image_files[idx]
                image = cv2.imread(str(img_path))
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # FIXED: Try multiple mask path strategies
                mask = None
                mask_paths_to_try = [
                    self.masks_dir / img_path.name,  # Same name
                    self.masks_dir / f"{img_path.stem}.png",  # Change extension to .png
                    self.masks_dir / img_path.stem / "mask.png",  # In subdirectory
                ]
                
                for mask_path in mask_paths_to_try:
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            break
                
                if mask is None:
                    # Try to find any mask with similar name
                    stem = img_path.stem
                    for mask_file in self.masks_dir.glob(f"{stem}*"):
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            break
                
                if mask is None:
                    raise ValueError(f"Could not find mask for {img_path.name}")
                
                # Convert mask to binary
                mask = (mask > 127).astype(np.float32)
                
                # Apply transforms
                transformed = self.transform(image=image, mask=mask)
                
                return transformed['image'], transformed['mask']
        
        # Create dataset
        print("\nPreparing dataset...")
        test_dataset = TestDataset(test_images_dir, test_masks_dir, CONFIG['img_size'])
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        print(f"✓ Dataset: {len(test_dataset)} images")
        
        # Evaluation functions
        def dice_coefficient(pred, target, smooth=1e-6):
            pred = (pred > 0.5).float()
            intersection = (pred * target).sum()
            return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        def iou_score(pred, target, smooth=1e-6):
            pred = (pred > 0.5).float()
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            return (intersection + smooth) / (union + smooth)
        
        # Run evaluation
        print("\nEvaluating...")
        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []
        inference_times = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)
                
                # Time inference
                start_time = time.time()
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000 / images.shape[0])
                
                # Calculate metrics per image
                for i in range(outputs.shape[0]):
                    pred = outputs[i]
                    target = masks[i]
                    
                    dice = dice_coefficient(pred, target).item()
                    iou = iou_score(pred, target).item()
                    
                    pred_binary = (pred > 0.5).float()
                    tp = (pred_binary * target).sum().item()
                    fp = (pred_binary * (1 - target)).sum().item()
                    fn = ((1 - pred_binary) * target).sum().item()
                    
                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    
                    all_dice.append(dice)
                    all_iou.append(iou)
                    all_precision.append(precision)
                    all_recall.append(recall)
        
        # Calculate metrics
        metrics = {
            'Model': 'UNet++ Segmentation',
            'Architecture': 'UNet++ (EfficientNet-B5)',
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            'Precision(Mask)': np.mean(all_precision),
            'Recall(Mask)': np.mean(all_recall),
            'Dice_Coefficient': np.mean(all_dice),
            'F1-Score(Mask)': np.mean(all_dice),
            'mAP50(Mask)': np.mean(all_iou),
            'mAP50-95(Mask)': np.mean(all_iou),
            
            'Speed_Inference(ms)': np.mean(inference_times),
            
            'Dice_Std': np.std(all_dice),
            'IoU_Mean': np.mean(all_iou),
            'IoU_Std': np.std(all_iou),
            'Num_Test_Images': len(test_dataset)
        }
        
        print("\n✅ UNet++ EVALUATION COMPLETE")
        print_metrics_summary(metrics)
        
        # Save results
        results_path = os.path.join(output_dir, 'unet_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Saved: {results_path}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ UNet++ evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_metrics_summary(metrics):
    """Print metrics summary"""
    print("\n📊 Metrics:")
    print("-" * 60)
    for key in ['Precision(Mask)', 'Recall(Mask)', 'F1-Score(Mask)', 
                'Dice_Coefficient', 'mAP50(Mask)', 'mAP50-95(Mask)']:
        if key in metrics:
            print(f"  {key:25s}: {metrics[key]:.4f}")
    if 'Speed_Inference(ms)' in metrics:
        print(f"  {'Speed':25s}: {metrics['Speed_Inference(ms)']:.2f} ms")
    print("-" * 60)

def calculate_improvement(unet_val, yolo_val):
    """Calculate improvement percentage"""
    if yolo_val == 0:
        return 0
    return ((unet_val - yolo_val) / yolo_val) * 100

def create_comparison_visualizations(yolo_metrics, unet_metrics, output_dir):
    """Create comparison charts"""
    print("\n📊 Creating visualizations...")
    
    key_metrics = [
        'Precision(Mask)', 'Recall(Mask)', 'F1-Score(Mask)',
        'Dice_Coefficient', 'mAP50(Mask)', 'mAP50-95(Mask)'
    ]
    
    available_metrics = [m for m in key_metrics if m in yolo_metrics and m in unet_metrics]
    
    if not available_metrics:
        print("⚠️  No common metrics for visualization")
        return []
    
    plots = []
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(available_metrics))
    width = 0.35
    
    yolo_vals = [yolo_metrics[m] for m in available_metrics]
    unet_vals = [unet_metrics[m] for m in available_metrics]
    
    ax.bar(x - width/2, yolo_vals, width, label='YOLO', color='#FF6B6B', alpha=0.8)
    ax.bar(x + width/2, unet_vals, width, label='UNet++', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('(Mask)', '') for m in available_metrics], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    for bars in [ax.containers[0], ax.containers[1]]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_bar_chart.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(plot_path)
    print(f"  ✓ Bar chart")
    
    # Improvement chart
    improvements = [calculate_improvement(unet_vals[i], yolo_vals[i]) 
                   for i in range(len(available_metrics))]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
    ax.barh([m.replace('(Mask)', '') for m in available_metrics], improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('UNet++ Improvement Over YOLO', fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, imp) in enumerate(zip(ax.containers[0], improvements)):
        width = bar.get_width()
        ax.text(width + (2 if width > 0 else -2), bar.get_y() + bar.get_height()/2.,
               f'{imp:+.1f}%',
               ha='left' if width > 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'improvement_chart.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots.append(plot_path)
    print(f"  ✓ Improvement chart")
    
    return plots

def generate_comparison_report(yolo_metrics, unet_metrics, output_dir):
    """Generate full comparison"""
    print("\n" + "="*80)
    print("📝 GENERATING COMPARISON REPORT")
    print("="*80)
    
    # Save CSV
    df = pd.DataFrame([yolo_metrics, unet_metrics])
    csv_path = os.path.join(output_dir, 'comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Calculate improvements
    key_metrics = [
        'Precision(Mask)', 'Recall(Mask)', 'F1-Score(Mask)',
        'Dice_Coefficient', 'mAP50(Mask)', 'mAP50-95(Mask)'
    ]
    
    improvements = {}
    wins_unet = 0
    wins_yolo = 0
    
    print("\n" + "-"*80)
    print("METRIC COMPARISON")
    print("-"*80)
    
    for metric in key_metrics:
        if metric in yolo_metrics and metric in unet_metrics:
            yolo_val = yolo_metrics[metric]
            unet_val = unet_metrics[metric]
            improvement = calculate_improvement(unet_val, yolo_val)
            
            improvements[metric] = {
                'YOLO': yolo_val,
                'UNet++': unet_val,
                'Improvement_%': improvement,
                'Absolute_Diff': unet_val - yolo_val
            }
            
            print(f"\n{metric}:")
            print(f"  YOLO:        {yolo_val:.4f}")
            print(f"  UNet++:      {unet_val:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
            
            if improvement > 0:
                print(f"  ✅ UNet++ WINS")
                wins_unet += 1
            elif improvement < 0:
                print(f"  ⚠️  YOLO wins")
                wins_yolo += 1
    
    # Save improvements
    improvements_path = os.path.join(output_dir, 'improvements.json')
    with open(improvements_path, 'w') as f:
        json.dump(improvements, f, indent=2)
    print(f"\n✓ Saved: {improvements_path}")
    
    # Create visualizations
    plots = create_comparison_visualizations(yolo_metrics, unet_metrics, output_dir)
    
    # Final verdict
    print("\n" + "="*80)
    print("🏆 FINAL VERDICT")
    print("="*80)
    total = len(improvements)
    print(f"UNet++ wins:  {wins_unet}/{total} ({wins_unet/total*100:.1f}%)")
    print(f"YOLO wins:    {wins_yolo}/{total} ({wins_yolo/total*100:.1f}%)")
    
    if wins_unet > wins_yolo:
        print(f"\n✅ CONCLUSION: UNet++ is SUPERIOR!")
        print(f"   Outperforms YOLO in {wins_unet}/{total} metrics")
    elif wins_yolo > wins_unet:
        print(f"\n⚠️  CONCLUSION: YOLO outperforms UNet++")
    else:
        print(f"\n= CONCLUSION: Models perform equally")
    
    avg_improvement = np.mean([v['Improvement_%'] for v in improvements.values()])
    print(f"\n📈 Average improvement: {avg_improvement:+.2f}%")
    print("="*80)
    
    return df, improvements, plots

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "#"*80)
    print("#" + " "*20 + "MODEL COMPARISON TOOL" + " "*37 + "#")
    print("#"*80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Evaluate YOLO
    yolo_metrics = evaluate_yolo_model(
        CONFIG['yolo_model_path'],
        CONFIG['data_yaml'],
        CONFIG['output_dir']
    )
    
    # Evaluate UNet++
    unet_metrics = evaluate_unet_model(
        CONFIG['unet_model_path'],
        CONFIG['test_images_dir'],
        CONFIG['test_masks_dir'],
        CONFIG['output_dir']
    )
    
    # Compare
    if yolo_metrics and unet_metrics:
        df, improvements, plots = generate_comparison_report(
            yolo_metrics,
            unet_metrics,
            CONFIG['output_dir']
        )
        
        print("\n" + "#"*80)
        print("✅ COMPLETE!")
        print("#"*80)
        print(f"\n📁 Results: {CONFIG['output_dir']}")
        print("\nFiles created:")
        print("  1. yolo_results.json")
        print("  2. unet_results.json")
        print("  3. comparison.csv")
        print("  4. improvements.json")
        print("  5. comparison_bar_chart.png")
        print("  6. improvement_chart.png")
        print("#"*80)
        
    else:
        print("\n❌ Evaluation failed - check errors above")

if __name__ == "__main__":
    main()