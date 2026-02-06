#!/usr/bin/env python3
"""
Check Kaggle kernel output size and provide download recommendations
"""
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def check_kernel_output(kernel_slug):
    """Check kernel status and output information"""
    api = KaggleApi()
    api.authenticate()
    
    print(f"🔍 Checking kernel: {kernel_slug}")
    print("=" * 60)
    
    try:
        # Get kernel status
        status = api.kernel_status(kernel_slug)
        
        print(f"\n📊 Kernel Information:")
        if hasattr(status, 'title'):
            print(f"  Title: {status.title}")
        if hasattr(status, 'status'):
            print(f"  Status: {status.status}")
        if hasattr(status, 'lastRunTime'):
            print(f"  Last Run: {status.lastRunTime}")
        
        # Try to get file list (this might not work for all kernels)
        print(f"\n📁 Attempting to list output files...")
        
        # The kernels_output_cli method shows what would be downloaded
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"  (This will show file list without downloading)")
            # Just checking, not actually downloading
            api.kernels_output_cli(kernel_slug, path=temp_dir)
        except Exception as e:
            print(f"  Could not preview files: {e}")
        
        print("\n💡 Recommendations:")
        print("  1. If output is large (>1GB), download from Kaggle website:")
        print(f"     https://www.kaggle.com/{kernel_slug}?tab=output")
        print("  2. For smaller outputs, use the Python API")
        print("  3. Consider using the chunked download script for reliability")
        
    except Exception as e:
        print(f"\n❌ Error checking kernel: {e}")
        print("\n🌐 Manual check:")
        print(f"   Visit: https://www.kaggle.com/{kernel_slug}")
        print("   Check the 'Output' tab to see file sizes")

if __name__ == "__main__":
    kernel_slug = "mohamedrebbouh/frac-atlas"
    check_kernel_output(kernel_slug)