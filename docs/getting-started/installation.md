# Installation

This guide walks you through installing SlicerSEEG on your system.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **3D Slicer** | 5.0+ | 5.6.2+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 2 GB | 5 GB |
| **GPU** | Optional | CUDA-compatible |

!!! note "GPU Acceleration"
    A CUDA-compatible GPU can significantly speed up the deep learning brain extraction step, but is not required.

---

## Step 1: Install 3D Slicer

If you don't have 3D Slicer installed:

1. Download from [slicer.org](https://download.slicer.org/)
2. Install following the platform-specific instructions
3. Launch 3D Slicer

---

## Step 2: Install Python Dependencies

SlicerSEEG requires several Python packages. Open the Python Interactor in Slicer:

**View → Python Interactor** (or press `Ctrl+3`)

### Automatic Installation (Recommended)

Copy and paste this code:

```python
import urllib.request
exec(urllib.request.urlopen(
    'https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/setup_dependencies.py'
).read().decode())
```

!!! info "Installation Time"
    This process takes 2-5 minutes depending on your internet connection. You'll see progress messages in the console.

### Manual Installation (Alternative)

If the automatic installation fails, install packages manually:

```python
import subprocess
import sys

packages = [
    'lightgbm',      # Confidence analysis
    'torch',         # Deep learning backend
    'monai',         # Medical imaging AI
    'networkx',      # Graph analysis
    'plotly',        # Interactive visualization
    'reportlab',     # PDF generation
    'pywavelets',    # Wavelet denoising
]

for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

print("Installation complete!")
```

---

## Step 3: Install the Extension

### Option A: Manual Installation

1. **Download** the [latest release](https://github.com/rociavl/SlicerSEEG/releases) from GitHub

2. **Extract** the archive to a location of your choice

3. **Add module path** in Slicer:
   - Go to **Edit → Application Settings → Modules**
   - Click the **>>** button under "Additional module paths"
   - Navigate to the `SEEG_ElectrodeLocalization` folder
   - Click **Select Folder**

4. **Restart** 3D Slicer

### Option B: Clone from GitHub

For developers who want the latest version:

```bash
git clone https://github.com/rociavl/SlicerSEEG.git
```

Then add the path to Slicer as described above.

---

## Step 4: Verify Installation

After restarting Slicer:

1. Open the **Module Selector** (dropdown in toolbar)
2. Navigate to **Segmentation → SEEG ElectrodeLocalization**
3. The module panel should appear

!!! success "Installation Successful"
    If you see the module panel with input selectors and the "Apply" button, installation is complete!

---

## Troubleshooting

### Module Not Found

If the module doesn't appear:

1. Check that the path in **Application Settings → Modules** is correct
2. Ensure you selected the `SEEG_ElectrodeLocalization` folder (not its parent)
3. Restart Slicer completely

### Import Errors

If you see Python import errors:

```python
# Check if packages are installed
import lightgbm
import torch
import monai
print("All packages imported successfully!")
```

If any import fails, reinstall that specific package.

### Memory Issues

For large CT scans (>512x512x512):

- Close other applications
- Increase Slicer's memory limit in **Edit → Application Settings → General**
- Consider downsampling the input volume

---

## Dependencies Reference

Full list of required packages:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0, <2.0 | Array operations |
| scipy | ≥1.7.0 | Scientific computing |
| pandas | ≥1.3.0 | Data manipulation |
| SimpleITK | ≥2.0.0 | Medical image I/O |
| scikit-image | ≥0.18.0 | Image processing |
| scikit-learn | ≥1.0.0 | Machine learning |
| lightgbm | ≥3.3.0 | Confidence analysis |
| torch | ≥1.10.0 | Deep learning |
| monai | ≥0.9.0 | Medical imaging AI |
| networkx | ≥2.6.0 | Graph analysis |
| matplotlib | ≥3.4.0 | Plotting |
| plotly | ≥5.0.0 | Interactive viz |
| pywavelets | ≥1.1.0 | Wavelet denoising |
| reportlab | ≥3.6.0 | PDF generation |

See [requirements.txt](https://github.com/rociavl/SlicerSEEG/blob/main/requirements.txt) for the complete list.

---

## Next Steps

Now that SlicerSEEG is installed, proceed to the [Quick Start Guide](quick-start.md) to process your first case.
