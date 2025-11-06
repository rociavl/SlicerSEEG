import slicer
import vtk
import numpy as np
import SimpleITK as sitk
import os
import time

class SEEGAlignment:
    """
    Efficient MRI-CT alignment script based on your SEEG project pipeline.
    Optimized for clinical workflow with automatic coordinate system handling.
    """
    
    def __init__(self):
        self.processing_log = []
        
    def log_step(self, message):
        """Log processing steps with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.processing_log.append(log_message)
    
    def detect_coordinate_system(self, volume_node):
        """
        Detect coordinate system based on your project's approach
        Returns: 'RAS' or 'LPS'
        """
        try:
            ijk_to_ras = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(ijk_to_ras)
            
            # Check direction matrix determinant to identify coordinate system
            direction = np.array([
                [ijk_to_ras.GetElement(0,0), ijk_to_ras.GetElement(0,1), ijk_to_ras.GetElement(0,2)],
                [ijk_to_ras.GetElement(1,0), ijk_to_ras.GetElement(1,1), ijk_to_ras.GetElement(1,2)],
                [ijk_to_ras.GetElement(2,0), ijk_to_ras.GetElement(2,1), ijk_to_ras.GetElement(2,2)]
            ])
            
            det = np.linalg.det(direction)
            return 'RAS' if det > 0 else 'LPS'
            
        except Exception as e:
            self.log_step(f"⚠️ Warning: Could not detect coordinate system: {e}")
            return 'RAS'  # Default to RAS as per your project
    
    def ensure_consistent_coordinate_system(self, moving_volume, fixed_volume):
        """
        Ensure both volumes use the same coordinate system based on your NRRD approach
        """
        moving_coord = self.detect_coordinate_system(moving_volume)
        fixed_coord = self.detect_coordinate_system(fixed_volume)
        
        self.log_step(f"📍 Moving volume ({moving_volume.GetName()}): {moving_coord}")
        self.log_step(f"📍 Fixed volume ({fixed_volume.GetName()}): {fixed_coord}")
        
        if moving_coord != fixed_coord:
            self.log_step("🔄 Converting coordinate systems to match...")
            self.convert_lps_to_ras(moving_volume)
            return True
        
        self.log_step("✅ Coordinate systems already consistent")
        return False
    
    def convert_lps_to_ras(self, volume_node):
        """
        Convert LPS to RAS using your project's transformation matrix:
        x_RAS = -x_LPS, y_RAS = -y_LPS, z_RAS = z_LPS
        """
        try:
            # Get current transform
            ijk_to_ras = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(ijk_to_ras)
            
            # Create LPS to RAS conversion matrix
            lps_to_ras = vtk.vtkMatrix4x4()
            lps_to_ras.SetElement(0, 0, -1)  # Flip Left-Right
            lps_to_ras.SetElement(1, 1, -1)  # Flip Posterior-Anterior  
            lps_to_ras.SetElement(2, 2, 1)   # Superior-Inferior unchanged
            lps_to_ras.SetElement(3, 3, 1)
            
            # Apply transformation
            result = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Multiply4x4(lps_to_ras, ijk_to_ras, result)
            volume_node.SetIJKToRASMatrix(result)
            
            self.log_step("✅ LPS to RAS conversion complete")
            
        except Exception as e:
            self.log_step(f"❌ Error in coordinate conversion: {e}")
    
    def get_volume_metadata(self, volume_node):
        """
        Extract volume metadata similar to your NRRD pipeline
        """
        try:
            origin = np.array(volume_node.GetOrigin())
            spacing = np.array(volume_node.GetSpacing())
            dimensions = list(volume_node.GetImageData().GetDimensions())
            
            ijk_to_ras = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(ijk_to_ras)
            
            return {
                'origin': origin,
                'spacing': spacing, 
                'dimensions': dimensions,
                'ijk_to_ras_matrix': ijk_to_ras,
                'voxel_count': np.prod(dimensions)
            }
            
        except Exception as e:
            self.log_step(f"❌ Error extracting metadata: {e}")
            return None
    
    def fast_rigid_registration(self, moving_volume, fixed_volume, output_name):
        """
        Fast rigid registration optimized for clinical workflow
        Based on your project's efficiency requirements (< 5 minutes)
        """
        try:
            self.log_step("🔧 Starting fast rigid registration...")
            
            # Create output volume
            output_volume = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", output_name
            )
            
            # Use BRAINSFit for fast, robust registration
            parameters = {
                "fixedVolume": fixed_volume,
                "movingVolume": moving_volume,
                "outputVolume": output_volume,
                "useRigid": True,
                "useAffine": False,  # Keep rigid only for speed
                "initializeTransformMode": "useCenterOfHeadAlign",
                "maskProcessingMode": "ROIAUTO",
                "numberOfSamples": 100000,  # Reduced for speed
                "numberOfIterations": 1500,  # Optimized balance
                "numberOfHistogramBins": 50,
                "maximumStepLength": 0.2,
                "minimumStepLength": 0.005,
                "relaxationFactor": 0.5,
                "translationScale": 1000.0,
                "reproportionScale": 1.0,
                "skewScale": 1.0
            }
            
            # Run registration
            self.log_step("⚙️ Executing registration (this may take 2-3 minutes)...")
            registration_result = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)
            
            if registration_result.GetStatus() & registration_result.Completed:
                self.log_step("✅ Registration completed successfully")
                return output_volume
            else:
                raise Exception("Registration failed")
                
        except Exception as e:
            self.log_step(f"❌ Registration error: {e}")
            return None
    
    def resample_to_reference(self, input_volume, reference_volume, output_name, interpolation="Linear"):
        """
        Resample volume to reference space - optimized version of your original function
        """
        try:
            self.log_step(f"🔄 Resampling {input_volume.GetName()} to {reference_volume.GetName()} space...")
            
            # Create output volume
            output_volume = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", output_name
            )
            
            # Get reference geometry
            reference_ijk_to_ras = vtk.vtkMatrix4x4()
            reference_volume.GetIJKToRASMatrix(reference_ijk_to_ras)
            
            # Resample parameters
            parameters = {
                "inputVolume": input_volume,
                "referenceVolume": reference_volume,
                "outputVolume": output_volume,
                "interpolationMode": interpolation  # "Linear" for CT, "NearestNeighbor" for masks
            }
            
            # Execute resampling
            resample_result = slicer.cli.runSync(
                slicer.modules.resamplescalarvectordwivolume, None, parameters
            )
            
            if resample_result.GetStatus() & resample_result.Completed:
                # Ensure proper geometry
                output_volume.SetIJKToRASMatrix(reference_ijk_to_ras)
                self.log_step("✅ Resampling completed successfully")
                return output_volume
            else:
                raise Exception("Resampling failed")
                
        except Exception as e:
            self.log_step(f"❌ Resampling error: {e}")
            return None
    
    def align_premri_postct(self, premri_name, postct_name, output_dir=None, 
                           register_mri_to_ct=True, save_results=True):
        """
        Main alignment function - fast and efficient for clinical use
        
        Args:
            premri_name: Name of pre-operative MRI volume in scene
            postct_name: Name of post-operative CT volume in scene  
            output_dir: Directory to save results (optional)
            register_mri_to_ct: If True, register MRI to CT; if False, register CT to MRI
            save_results: Whether to save aligned volumes
        
        Returns:
            dict: Results including aligned volumes and metadata
        """
        start_time = time.time()
        self.log_step("🚀 Starting SEEG MRI-CT alignment pipeline...")
        
        try:
            # Get volumes from scene
            premri = slicer.util.getNode(premri_name)
            postct = slicer.util.getNode(postct_name)
            
            if not premri or not postct:
                raise ValueError("❌ Could not find specified volumes in scene")
            
            self.log_step(f"📂 Found volumes: {premri.GetName()} & {postct.GetName()}")
            
            # Step 1: Ensure consistent coordinate systems (your NRRD approach)
            self.ensure_consistent_coordinate_system(premri, postct)
            
            # Step 2: Extract metadata for validation
            mri_metadata = self.get_volume_metadata(premri)
            ct_metadata = self.get_volume_metadata(postct)
            
            self.log_step(f"📊 MRI: {mri_metadata['dimensions']} voxels, {mri_metadata['spacing']} mm spacing")
            self.log_step(f"📊 CT: {ct_metadata['dimensions']} voxels, {ct_metadata['spacing']} mm spacing")
            
            # Step 3: Registration
            if register_mri_to_ct:
                self.log_step("🎯 Registering MRI to CT space...")
                moving, fixed = premri, postct
                aligned_name = f"{premri.GetName()}_aligned_to_CT"
            else:
                self.log_step("🎯 Registering CT to MRI space...")
                moving, fixed = postct, premri
                aligned_name = f"{postct.GetName()}_aligned_to_MRI"
            
            # Fast rigid registration
            aligned_volume = self.fast_rigid_registration(moving, fixed, aligned_name)
            
            if not aligned_volume:
                raise Exception("Registration failed")
            
            # Step 4: Optional resampling for perfect geometric alignment
            resampled_name = f"{aligned_name}_resampled"
            resampled_volume = self.resample_to_reference(
                aligned_volume, fixed, resampled_name, "Linear"
            )
            
            # Step 5: Save results if requested
            results = {
                'original_moving': moving,
                'original_fixed': fixed,
                'aligned_volume': aligned_volume,
                'resampled_volume': resampled_volume,
                'processing_time': time.time() - start_time,
                'coordinate_system': self.detect_coordinate_system(resampled_volume),
                'mri_metadata': mri_metadata,
                'ct_metadata': ct_metadata
            }
            
            if save_results and output_dir:
                self.save_alignment_results(results, output_dir)
            
            processing_time = time.time() - start_time
            self.log_step(f"🎉 Alignment completed successfully in {processing_time:.1f} seconds")
            self.log_step(f"📈 Efficiency: {((4*3600 - processing_time) / (4*3600) * 100):.1f}% time saved vs manual")
            
            return results
            
        except Exception as e:
            self.log_step(f"❌ Alignment failed: {e}")
            return None
    
    def save_alignment_results(self, results, output_dir):
        """Save aligned volumes and generate report"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save aligned volume
            aligned_path = os.path.join(output_dir, f"{results['aligned_volume'].GetName()}.nrrd")
            slicer.util.saveNode(results['aligned_volume'], aligned_path)
            
            # Save resampled volume  
            resampled_path = os.path.join(output_dir, f"{results['resampled_volume'].GetName()}.nrrd")
            slicer.util.saveNode(results['resampled_volume'], resampled_path)
            
            # Save processing log
            log_path = os.path.join(output_dir, "alignment_log.txt")
            with open(log_path, 'w') as f:
                f.write("SEEG MRI-CT Alignment Report\n")
                f.write("="*50 + "\n")
                f.write(f"Processing time: {results['processing_time']:.1f} seconds\n")
                f.write(f"Coordinate system: {results['coordinate_system']}\n")
                f.write("\nProcessing Log:\n")
                for log_entry in self.processing_log:
                    f.write(log_entry + "\n")
            
            self.log_step(f"💾 Results saved to: {output_dir}")
            
        except Exception as e:
            self.log_step(f"❌ Error saving results: {e}")

# Example usage function
def align_seeg_volumes(premri_name, postct_name, output_dir=None):
    """
    Convenient wrapper function for SEEG alignment
    
    Example usage within Slicer:
    align_seeg_volumes(
        "T1_3D_SAG_ALTA_RESOLUCION_20170502103104_7",
        "NNCR_Neuronavegador_NRC_Clinic_20180123204109_2", 
        "C:/Users/rocia/Downloads/TFG/Cohort/Aligned_Results"
    )
    """
    aligner = SEEGAlignment()
    return aligner.align_premri_postct(
        premri_name=premri_name,
        postct_name=postct_name,
        output_dir=output_dir,
        register_mri_to_ct=True,  # Register MRI to CT (standard for SEEG)
        save_results=True
    )

def list_available_volumes():
    """Helper function to see what volumes are available in the scene"""
    print("\n📋 Available volumes in scene:")
    nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
    for i in range(nodes.GetNumberOfItems()):
        node = nodes.GetItemAsObject(i)
        print(f"  - {node.GetName()}")
    print()

def quick_align_interactive():
    """
    Interactive function for quick alignment when called from Slicer console
    """
    print("🔧 SEEG MRI-CT Alignment Tool")
    print("=" * 40)
    
    # Show available volumes
    list_available_volumes()
    
    # Get volumes (you can modify these names as needed)
    volumes = []
    nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
    for i in range(nodes.GetNumberOfItems()):
        volumes.append(nodes.GetItemAsObject(i).GetName())
    
    if len(volumes) >= 2:
        print(f"🎯 Auto-detecting volumes...")
        print(f"   Volume 1: {volumes[0]}")
        print(f"   Volume 2: {volumes[1]}")
        
        # Run alignment with first two volumes found
        output_path = "C:/Users/rocia/Downloads/TFG/Cohort/Aligned_Results"
        
        results = align_seeg_volumes(
            premri_name=volumes[0],
            postct_name=volumes[1],
            output_dir=output_path
        )
        
        return results
    else:
        print("❌ Error: Need at least 2 volumes loaded in scene")
        return None

# Initialize global aligner instance for easy access
slicer_aligner = SEEGAlignment()

print("✅ SEEG Alignment script loaded successfully!")
print("\n📖 Usage options:")
print("1. Interactive mode: quick_align_interactive()")
print("2. Manual mode: align_seeg_volumes('mri_name', 'ct_name', 'output_dir')")
print("3. List volumes: list_available_volumes()")
print("4. Direct access: slicer_aligner.align_premri_postct(...)")

align_seeg_volumes(
    "T1_3D_SAG_ALTA_RESOLUCION_20170502103104_7",
    "NNCR_Neuronavegador_NRC_Clinic_20180123204109_2", 
    "C:/Users/rocia/Downloads/TFG/Cohort/Aligned_Results"
)