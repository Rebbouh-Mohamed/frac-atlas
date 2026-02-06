import os

# Define base paths
BASE_DIR = "/home/rebbouh/data_segm"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

# Create directory structure
directories = [
    os.path.join(ANALYSIS_DIR, "scripts"),
    os.path.join(ANALYSIS_DIR, "results"),
    os.path.join(ANALYSIS_DIR, "plots"),
    os.path.join(ANALYSIS_DIR, "report"),
    os.path.join(ANALYSIS_DIR, "config_comparison")
]

print("\n" + "="*70)
print("CREATING ANALYSIS DIRECTORY STRUCTURE")
print("="*70 + "\n")

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Created: {directory}")

# Save config files for reference
config_v7 = """task: segment
mode: train
model: yolo11m-seg.pt
data: /kaggle/working/data.yaml
epochs: 100
batch: 16
imgsz: 800
# KEY: Image size 800, 100 epochs
# This is our reproduction attempt
"""

config_v8 = """task: segment
mode: train
model: yolo11m-seg.pt
data: /kaggle/working/data.yaml
epochs: 120
batch: 16
imgsz: 640
# KEY: Image size 640, 120 epochs  
# This is our improved version with more training
"""

config_dir = os.path.join(ANALYSIS_DIR, "config_comparison")
with open(os.path.join(config_dir, "v7_config_summary.txt"), 'w') as f:
    f.write(config_v7)
    
with open(os.path.join(config_dir, "v8_config_summary.txt"), 'w') as f:
    f.write(config_v8)

print(f"\n✓ Saved configuration summaries to: {config_dir}")
print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)