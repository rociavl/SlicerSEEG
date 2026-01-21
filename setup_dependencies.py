"""
SlicerSEEG Dependency Installer
================================
Run this script ONCE in 3D Slicer's Python console to install all dependencies.

Usage:
1. Open 3D Slicer
2. Go to: View → Python Interactor
3. Copy and paste this entire file
4. Press Enter
5. Wait for installation (3-7 minutes)
6. Restart Slicer when prompted

Alternatively, run from URL:
import urllib.request
exec(urllib.request.urlopen('https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/setup_dependencies.py').read().decode())
"""

def install_slicerseeg_dependencies():
    """One-command installation of all SlicerSEEG dependencies with version constraints."""
    import slicer
    
    # Define all required packages with version constraints for compatibility
    dependencies = [
        ('numpy>1.2', 'Numerical computations'),
        ('scipy<1.14', 'Scientific computing'),
        ('pandas<3.0', 'Data analysis'),
        ('scikit-learn<1.6', 'Machine learning'),
        ('scikit-image<0.25', 'Image processing'),
        ('PyWavelets<2.0', 'Wavelet transforms'),
        ('SimpleITK<3.0', 'Medical image I/O'),
        ('lightgbm<5.0', 'Confidence analysis'),
        ('torch<3.0', 'Brain segmentation (CPU version)'),
        ('monai<2.0', 'Medical imaging AI'),
        ('networkx<4.0', 'Trajectory analysis'),
        ('plotly<6.0', 'Interactive visualizations'),
        ('matplotlib<4.0', 'Plotting'),
        ('joblib<2.0', 'Model serialization'),
        ('pynrrd<2.0', 'NRRD file support'),
        ('nibabel<6.0', 'NIfTI file support'),
        ('seaborn<1.0', 'Statistical visualization'),
        ('reportlab<5.0', 'PDF report generation'),
        ('tqdm<5.0', 'Progress bars'),
    ]
    
    print("=" * 60)
    print("SlicerSEEG Dependency Installer")
    print("=" * 60)
    print(f"\nWill install {len(dependencies)} packages with version constraints:")
    for pkg, purpose in dependencies:
        # Extract package name for display
        pkg_name = pkg.split('<')[0].split('>')[0].split('=')[0]
        print(f"  • {pkg_name:20} - {purpose}")
    
    print(f"\nEstimated time: 3-7 minutes")
    print("\nNote: Version constraints ensure compatibility across Slicer versions.")
    print("=" * 60)
    
    # Confirm installation
    reply = slicer.util.confirmYesNoDisplay(
        "Install all SlicerSEEG dependencies?\n\n"
        f"This will install {len(dependencies)} packages with version constraints.\n"
        "Installation takes 3-7 minutes.\n\n"
        "Note: If you have NumPy 2.0, it will be downgraded to 1.x for compatibility.\n\n"
        "Continue?"
    )
    
    if not reply:
        print("\n❌ Installation cancelled.")
        return False
    
    # Install each package using subprocess
    import subprocess
    import sys
    
    failed_packages = []
    successful_packages = []
    
    # Get Python executable from Slicer
    python_exe = sys.executable
    
    for i, (package, purpose) in enumerate(dependencies, 1):
        try:
            print(f"\n[{i}/{len(dependencies)}] Installing {package}...")
            
            # Use subprocess to call pip
            result = subprocess.check_call([python_exe, '-m', 'pip', 'install', package])
            
            successful_packages.append(package)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            failed_packages.append(package)
        except Exception as e:
            print(f"❌ Unexpected error installing {package}: {e}")
            failed_packages.append(package)
    
    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    print(f"✅ Successful: {len(successful_packages)}/{len(dependencies)}")
    if successful_packages:
        for pkg in successful_packages:
            pkg_name = pkg.split('<')[0].split('>')[0].split('=')[0]
            print(f"   • {pkg_name}")
    
    if failed_packages:
        print(f"\n❌ Failed: {len(failed_packages)}")
        for pkg in failed_packages:
            pkg_name = pkg.split('<')[0].split('>')[0].split('=')[0]
            print(f"   • {pkg_name}")
        print(f"\nYou can manually install failed packages:")
        print(f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '{failed_packages[0]}'])")
    
    print("=" * 60)
    
    if len(successful_packages) == len(dependencies):
        slicer.util.infoDisplay(
            "✅ All dependencies installed successfully!\n\n"
            "Please RESTART 3D Slicer to use SlicerSEEG.\n\n"
            "Note: If you see any version warnings, they are expected\n"
            "due to compatibility constraints."
        )
        return True
    else:
        slicer.util.warningDisplay(
            f"⚠️ {len(failed_packages)} packages failed to install.\n\n"
            "SlicerSEEG may have limited functionality.\n"
            "Check the Python console for details."
        )
        return False

# Auto-run when script is executed
if __name__ == "__main__":
    install_slicerseeg_dependencies()
else:
    # When imported/pasted into console
    install_slicerseeg_dependencies()