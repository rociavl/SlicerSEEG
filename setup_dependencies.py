"""
SlicerSEEG Dependency Installer
================================
Run this script ONCE in 3D Slicer's Python console to install all dependencies.

Usage:
1. Open 3D Slicer
2. Go to: View → Python Interactor
3. Copy and paste this entire file
4. Press Enter
5. Wait for installation (2-5 minutes)
6. Restart Slicer when prompted

Alternatively, if you have this file saved:
exec(open('/path/to/setup_dependencies.py').read())
"""

def install_slicerseeg_dependencies():
    """One-command installation of all SlicerSEEG dependencies."""
    import slicer
    
    # Define all required packages
    dependencies = [
        ('lightgbm', 'Confidence analysis'),
        ('torch', 'Brain segmentation (CPU version)'),
        ('monai', 'Brain segmentation'),
        ('networkx', 'Trajectory analysis'),
        ('plotly', 'Interactive visualizations'),
        ('tqdm', 'Progress bars'),
    ]
    
    print("=" * 60)
    print("SlicerSEEG Dependency Installer")
    print("=" * 60)
    print(f"\nWill install {len(dependencies)} packages:")
    for pkg, purpose in dependencies:
        print(f"  • {pkg:15} - {purpose}")
    
    print(f"\nEstimated time: 2-5 minutes")
    print("=" * 60)
    
    # Confirm installation
    reply = slicer.util.confirmYesNoDisplay(
        "Install all SlicerSEEG dependencies?\n\n"
        f"This will install {len(dependencies)} packages.\n"
        "Installation takes 2-5 minutes.\n\n"
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
            print(f"   • {pkg}")
    
    if failed_packages:
        print(f"\n❌ Failed: {len(failed_packages)}")
        for pkg in failed_packages:
            print(f"   • {pkg}")
        print(f"\nYou can manually install failed packages:")
        print(f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '{failed_packages[0]}'])")
    
    print("=" * 60)
    
    if len(successful_packages) == len(dependencies):
        slicer.util.infoDisplay(
            "✅ All dependencies installed successfully!\n\n"
            "Please RESTART 3D Slicer to use SlicerSEEG."
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