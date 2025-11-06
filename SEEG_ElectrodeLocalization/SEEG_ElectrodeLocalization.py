import logging
import os
from typing import Annotated, Optional
import sys
import csv
import slicer.util
import vtk
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)

# Add the Brain_mask_methods directory to the Python path
module_dir = os.path.dirname(__file__)
brain_mask_methods_dir = os.path.join(module_dir, "Brain_mask_methods")
if brain_mask_methods_dir not in sys.path:
    sys.path.append(brain_mask_methods_dir)

# Import the BrainMaskExtractor from the correct module
from brain_mask_extractor_otsu import BrainMaskExtractor 

# Add the Threshold_mask directory to the Python path
threshold_mask_dir = os.path.join(module_dir, "Threshold_mask")
if threshold_mask_dir not in sys.path:
    sys.path.append(threshold_mask_dir)

global_masking_dir = os.path.join(module_dir, "Masks_ensemble")
if global_masking_dir not in sys.path:
    sys.path.append(global_masking_dir)

# Import the EnhancedMaskSelector
from masks_fusion_top import EnhancedMaskSelector

# Import CTPEnhancer from enhance_ctp
from Threshold_mask.ctp_enhancer import CTPEnhancer

centroids_pipeline_dir = os.path.join(module_dir, "Centroids_pipeline")
if centroids_pipeline_dir not in sys.path:
    sys.path.append(centroids_pipeline_dir)

from centroids_feature_extraction import extract_all_target_features

# TRAJECTORY PART
# === ADD THIS NEW SECTION FOR TRAJECTORY ANALYSIS ===
electrode_path_dir = os.path.join(module_dir, "Electrode_path")
if electrode_path_dir not in sys.path:
    sys.path.append(electrode_path_dir)

# Import trajectory analysis functions
try:
    from test_slicer import (
        integrated_trajectory_analysis,
        calculate_trajectory_scores,
        create_interactive_annotation_report,
        create_basic_3d_plot,
        analyze_both_hemispheres_separately,
        apply_hemisphere_splitting_to_results,
        adaptive_clustering_parameters
    )
    TRAJECTORY_ANALYSIS_AVAILABLE = True
    print("✅ Trajectory analysis module loaded successfully")
except ImportError as e:
    print(f"⚠️ Trajectory analysis module not available: {e}")
    TRAJECTORY_ANALYSIS_AVAILABLE = False
 
 ###--Bilateral detection and splitting function--###
def detect_and_split_bilateral_electrodes(electrode_coords, min_per_hemisphere=2):
    """
    Simple bilateral detection and splitting function.
    Returns: (is_bilateral, left_coords, right_coords, hemisphere_info)
    """
    if len(electrode_coords) == 0:
        return False, None, None, {}
    
    # Split by RAS X coordinate (left = x<0, right = x>0)
    left_mask = electrode_coords[:, 0] < 0
    right_mask = electrode_coords[:, 0] > 0
    
    left_coords = electrode_coords[left_mask]
    right_coords = electrode_coords[right_mask]
    
    is_bilateral = (len(left_coords) >= min_per_hemisphere and 
                   len(right_coords) >= min_per_hemisphere)
    
    hemisphere_info = {
        'total_electrodes': len(electrode_coords),
        'left_count': len(left_coords),
        'right_count': len(right_coords),
        'is_bilateral': is_bilateral,
        'processing_mode': 'bilateral' if is_bilateral else 'unified'
    }
    
    logging.info(f"Electrode distribution: {len(left_coords)} left, {len(right_coords)} right. "
                f"Mode: {'bilateral' if is_bilateral else 'unified'}")
    
    return is_bilateral, left_coords, right_coords, hemisphere_info

# Define path to the model
MODEL_PATH = os.path.join(module_dir, "models", "random_forest_modelP1.joblib")
CONFIDENCE_MODEL_PATH= os.path.join(module_dir, "models", "patient_leave_one_out_ensemble.joblib")

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from slicer import vtkMRMLScalarVolumeNode
import qt

#
# Model confidence import 
#

# === EMBEDDED CONFIDENCE ANALYSIS ===

try:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from lightgbm import LGBMRegressor
    import pandas as pd
    import numpy as np
    import joblib
    import warnings
    CONFIDENCE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Confidence analysis dependencies missing: {e}")
    CONFIDENCE_AVAILABLE = False

# ===  WORKING CLASSES  ===

class PatientEnsemblePipeline:
    def __init__(self, numerical_features=None, categorical_features=None, test_size=0.2, random_state=42):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.model_features = self.numerical_features + self.categorical_features
        self.patient_models = {}
        self.model_weights = {}
        self.test_size = test_size
        self.random_state = random_state

    def predict_leave_one_out(self, test_df, exclude_patient_id=None):
        """Make predictions using leave-one-out weighted ensemble"""
        if not self.patient_models:
            raise ValueError("No models trained.")

        predictions = {}
        for patient_id, model in self.patient_models.items():
            if patient_id == exclude_patient_id:
                continue

            X_test = test_df[model.numerical_features + model.categorical_features]
            preds = model.model.predict(X_test)
            predictions[patient_id] = preds

        if not predictions:
            raise ValueError("No models available for prediction")

        weighted_preds = np.zeros(len(test_df))
        total_weight = 0

        for patient_id, preds in predictions.items():
            weight = self.model_weights.get(patient_id, 1.0)
            weighted_preds += weight * preds
            total_weight += weight

        if total_weight > 0:
            weighted_preds /= total_weight

        return weighted_preds

class CentroidConfidencePipeline:
    """Minimal class definition for joblib loading"""
    def __init__(self, numerical_features=None, categorical_features=None, test_size=0.2, random_state=42):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        self.gt_tree = None
        self.preprocessing_time = 0
        self.model_training_time = 0
        self.total_training_time = 0
        self.test_metrics = None
        self.feature_names = None
        self.best_params = None

# === FORCE CLASSES INTO __main__ MODULE ===

def force_classes_into_main():
    """Force our classes into __main__ module where joblib expects them"""
    import sys
    import types
    
    # Get or create __main__ module
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = types.ModuleType('__main__')
    
    main_module = sys.modules['__main__']
    
    # Force our classes into __main__
    main_module.PatientEnsemblePipeline = PatientEnsemblePipeline
    main_module.CentroidConfidencePipeline = CentroidConfidencePipeline
    
    # Also add them to globals() for extra safety
    import builtins
    builtins.PatientEnsemblePipeline = PatientEnsemblePipeline
    builtins.CentroidConfidencePipeline = CentroidConfidencePipeline
    
    print("✅ Classes forced into __main__ module")

# Call this immediately
force_classes_into_main()


# ===  WORKING FUNCTIONS  ===

def extract_centroids_from_mask(mask_file_path):
    """Extract electrode centroid coordinates from a mask file."""
    try:
        import SimpleITK as sitk
        from skimage.measure import label, regionprops
        
        # Load mask
        mask_image = sitk.ReadImage(mask_file_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        
        # Find connected components
        labeled_mask = label(mask_array > 0)
        regions = regionprops(labeled_mask)
        
        # Extract centroids
        electrode_coords = []
        for region in regions:
            # Get centroid in voxel coordinates (z, y, x)
            centroid_voxel = region.centroid
            
            # Convert to physical coordinates (x, y, z)
            idx_coord = (float(centroid_voxel[2]), float(centroid_voxel[1]), float(centroid_voxel[0]))
            physical_point = mask_image.TransformContinuousIndexToPhysicalPoint(idx_coord)
            
            # Convert LPS to RAS
            ras_coord = (-physical_point[0], -physical_point[1], physical_point[2])
            electrode_coords.append(ras_coord)
        
        logging.info(f"Extracted {len(electrode_coords)} centroids from mask")
        return np.array(electrode_coords)
        
    except Exception as e:
        logging.error(f"Error extracting centroids: {str(e)}")
        return np.array([])

def create_electrode_csv(electrode_coords, volume_name, output_file):
    """Create a temporary CSV file with electrode coordinates for feature extraction."""
    import pandas as pd
    
    df = pd.DataFrame({
        'RAS_X': electrode_coords[:, 0],
        'RAS_Y': electrode_coords[:, 1], 
        'RAS_Z': electrode_coords[:, 2],
        'Patient ID': volume_name,
        'Pixel Count': 1  # Default value required by feature extraction
    })
    
    df.to_csv(output_file, index=False)
    logging.info(f"Created temporary electrode CSV: {output_file}")
    return output_file


def run_embedded_confidence_analysis(brain_mask_file, top_mask_file, enhanced_ct_file, volume_name, confidence_dir, model_path):
    """
    Complete confidence analysis using your existing feature extraction + working model
    """
    logging.info("🔧 Starting embedded confidence analysis...")
    
    if not CONFIDENCE_AVAILABLE:
        raise ImportError("Confidence analysis dependencies not available")
    
    try:
        # Extract centroids from top mask 
        logging.info("📍 Step 1: Extracting centroids...")
        electrode_coords = extract_centroids_from_mask(top_mask_file)
            # ADD THESE 3 LINES:
        is_bilateral, left_coords, right_coords, hemi_info = detect_and_split_bilateral_electrodes(electrode_coords)
        if is_bilateral:
            logging.info(f"Bilateral configuration detected: processing {hemi_info['left_count']} left + {hemi_info['right_count']} right electrodes")
    
        
        if len(electrode_coords) == 0:
            raise ValueError("No electrode centroids found in mask")
        
        #  temporary CSV for the feature extraction function
        logging.info("🔬 Step 2: Creating temporary electrode CSV...")
        temp_csv_file = os.path.join(confidence_dir, "temp_electrode_coords.csv")
        temp_df = pd.DataFrame({
            'RAS_X': electrode_coords[:, 0],
            'RAS_Y': electrode_coords[:, 1], 
            'RAS_Z': electrode_coords[:, 2],
            'Patient ID': volume_name,
            'Pixel Count': 1
        })
        temp_df.to_csv(temp_csv_file, index=False)
        
        # feature extraction function
        logging.info("🔬 Step 3: Running your feature extraction function...")
        
        
        features_df = extract_all_target_features(
            results_file=temp_csv_file,
            ct_file=enhanced_ct_file,
            roi_file_path=brain_mask_file,
            patient_id=volume_name,
            mask_id="top_mask_1",
            output_dir=confidence_dir
        )
        
        features_csv = os.path.join(confidence_dir, f"target_features_{volume_name}_top_mask_1.csv")
        logging.info(f"✅ Features saved to: {features_csv}")
        
        # Step 4
        logging.info("🤖 Step 4: Running prediction with your working model...")
        predictions_csv = os.path.join(confidence_dir, f"confidence_predictions_{volume_name}.csv")
        
        # Use your exact working function (now embedded)
        confidence_results = embedded_predict_electrode_confidence(
            feature_csv_path=features_csv,
            model_path=model_path,
            patient_id=volume_name,
            output_path=predictions_csv
        )
        
        # Step 5: Calculate statistics
        logging.info("📊 Step 5: Calculating statistics...")
        total_electrodes = len(confidence_results)
        high_confidence = np.sum(confidence_results['Ensemble_Confidence'] >= 0.8)
        medium_confidence = np.sum((confidence_results['Ensemble_Confidence'] >= 0.6) & 
                                 (confidence_results['Ensemble_Confidence'] < 0.8))
        low_confidence = np.sum(confidence_results['Ensemble_Confidence'] < 0.6)
        top_confidence = confidence_results['Ensemble_Confidence'].max()
        avg_confidence = confidence_results['Ensemble_Confidence'].mean()
        
        # Clean up temporary file
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)
        
        logging.info("✅ Embedded confidence analysis completed successfully!")
        
        return {
            'confidence_results': confidence_results,
            'total_electrodes': total_electrodes,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'top_confidence': top_confidence,
            'avg_confidence': avg_confidence,
            'predictions_csv': predictions_csv
        }
        
    except Exception as e:
        logging.error(f"❌ Embedded confidence analysis failed: {e}")
        raise





def create_confidence_summary(confidence_results, summary_file, volume_name, 
                             total_electrodes, high_confidence, medium_confidence, 
                             top_confidence, avg_confidence):
    """Create a simplified text summary of confidence analysis."""
    from datetime import datetime
    
    # Get top 10 electrodes
    top_10 = confidence_results.head(10)
    
    summary_content = f"""
ELECTRODE CONFIDENCE ANALYSIS SUMMARY
====================================
Patient/Volume: {volume_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
- Total electrode candidates: {total_electrodes}
- High confidence (≥80%): {high_confidence} electrodes
- Medium confidence (60-80%): {medium_confidence} electrodes
- Low confidence (<60%): {total_electrodes - high_confidence - medium_confidence} electrodes
- Average confidence: {avg_confidence:.3f}
- Top confidence score: {top_confidence:.3f}

TOP 10 ELECTRODE CANDIDATES:
Rank | RAS Coordinates (X, Y, Z) | Confidence | Recommendation
-----|---------------------------|------------|---------------
"""
    
    for _, row in top_10.iterrows():
        conf = row['Ensemble_Confidence']
        if conf >= 0.7:
            recommendation = "Excellent"
        elif conf >= 0.4:
            recommendation = "Good"
        else:
            recommendation = "Caution"
            
        summary_content += f"{int(row['Confidence_Rank']):4d} | ({row['RAS_X']:6.1f}, {row['RAS_Y']:6.1f}, {row['RAS_Z']:6.1f}) | {conf:8.3f} | {recommendation}\n"
    
    summary_content += f"""
CLINICAL RECOMMENDATIONS:
- High Confidence (≥70%): Primary electrode placement candidates
- Medium Confidence (30-70%): Secondary candidates, consider additional verification  
- Low Confidence (<30%): Use with caution, require additional evaluation

FILES GENERATED:
- Features: target_features_{volume_name}_top_mask_1.csv
- Predictions: confidence_predictions_{volume_name}.csv
- Summary: confidence_summary_{volume_name}.txt
"""
    
    # Save summary
    with open(summary_file, 'w') as f:
        f.write(summary_content)
    
    logging.info(f"Confidence summary saved: {summary_file}")
    return summary_file


def embedded_load_model_and_predict(csv_path, model_path, exclude_patient="P8"):
    """Your exact working function with forced class registration"""
    print(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} electrodes")

    # Fix categorical columns (CRITICAL for model compatibility)
    categorical_columns = ['Hemisphere', 'has_neighbors']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Fixed {col} as categorical")

    # CRITICAL: Force classes into __main__ before loading model
    print("Forcing classes into __main__ before model loading...")
    force_classes_into_main()
    
    # Load trained model
    print(f"Loading model from: {model_path}")
    try:
        ensemble = joblib.load(model_path)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
        # Try alternative loading approaches
        print("🔄 Trying alternative loading method...")
        
        # Method 1: Custom unpickler
        try:
            import pickle
            
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == '__main__':
                        if name == 'PatientEnsemblePipeline':
                            return PatientEnsemblePipeline
                        elif name == 'CentroidConfidencePipeline':
                            return CentroidConfidencePipeline
                    return super().find_class(module, name)
            
            with open(model_path, 'rb') as f:
                ensemble = CustomUnpickler(f).load()
            print("✅ Model loaded with custom unpickler")
            
        except Exception as e2:
            print(f"❌ Custom unpickler failed: {e2}")
            
            # Method 2: Try loading with different module context
            try:
                import sys
                original_main = sys.modules.get('__main__')
                
                # Create a fake __main__ module with our classes
                fake_main = types.ModuleType('__main__')
                fake_main.PatientEnsemblePipeline = PatientEnsemblePipeline
                fake_main.CentroidConfidencePipeline = CentroidConfidencePipeline
                fake_main.__file__ = '__main__'
                
                sys.modules['__main__'] = fake_main
                
                ensemble = joblib.load(model_path)
                print("✅ Model loaded with fake __main__")
                
                # Restore original __main__
                if original_main:
                    sys.modules['__main__'] = original_main
                
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                raise e  # Re-raise original error

    # Make predictions
    print(f"Making predictions (excluding {exclude_patient})...")
    confidence_scores = ensemble.predict_leave_one_out(df, exclude_patient_id=exclude_patient)

    # Add predictions to dataframe
    df['Ensemble_Confidence'] = confidence_scores
    df['Binary_Prediction'] = (confidence_scores >= 0.5).astype(int)

    # Sort by confidence (highest first)
    df = df.sort_values('Ensemble_Confidence', ascending=False).reset_index(drop=True)
    df['Confidence_Rank'] = range(1, len(df) + 1)

    # Print summary
    print(f"Predictions complete!")
    print(f"  Top confidence score: {confidence_scores.max():.4f}")
    print(f"  Mean confidence score: {confidence_scores.mean():.4f}")
    print(f"  High confidence predictions: {np.sum(confidence_scores >= 0.5)}/{len(confidence_scores)}")

    return df

def embedded_predict_electrode_confidence(feature_csv_path, model_path, patient_id="P8", output_path=None):
    """Your exact working function"""
    try:
        results_df = embedded_load_model_and_predict(
            csv_path=feature_csv_path,
            model_path=model_path,
            exclude_patient=patient_id
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

        # Show top predictions
        print(f"\nTop 5 predictions:")
        top_5 = results_df[['Confidence_Rank', 'RAS_X', 'RAS_Y', 'RAS_Z', 'Ensemble_Confidence']].head(5)
        for _, row in top_5.iterrows():
            print(f"  #{int(row['Confidence_Rank'])}: "
                  f"({row['RAS_X']:.1f}, {row['RAS_Y']:.1f}, {row['RAS_Z']:.1f}) "
                  f"confidence: {row['Ensemble_Confidence']:.4f}")

        return results_df

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

# === TRAJECTORY ANALYSIS FUNCTIONS ===

def run_trajectory_analysis_on_markups(markup_node, trajectory_ids=None, output_dir=None, volume_name="analysis"):
    """
    Run trajectory analysis on selected markup points with bilateral support.
    
    Args:
        markup_node: vtkMRMLMarkupsFiducialNode with electrode points
        trajectory_ids: List of trajectory IDs to analyze (None = analyze all)
        output_dir: Directory to save results
        volume_name: Name for output files
    
    Returns:
        dict: Analysis results
    """
    if not TRAJECTORY_ANALYSIS_AVAILABLE:
        raise ImportError("Trajectory analysis module not available")
    
    try:
        logging.info("🔧 Starting trajectory analysis on selected markups...")
        
        # Extract coordinates from markup node
        coords_list = []
        point_ids = []
        
        num_points = markup_node.GetNumberOfControlPoints()
        if num_points < 6:
            return {
                'success': False,
                'reason': f'Insufficient points ({num_points}). Need at least 6 for trajectory analysis.',
                'total_points': num_points
            }
        
        # Get all points from markup
        for i in range(num_points):
            pos = [0, 0, 0]
            markup_node.GetNthControlPointPosition(i, pos)
            coords_list.append(pos)
            point_ids.append(i)
        
        coords_array = np.array(coords_list)
        logging.info(f"Extracted {len(coords_array)} electrode coordinates from markup")
        
        # Check for bilateral electrode configuration
        is_bilateral, left_coords, right_coords, hemi_info = detect_and_split_bilateral_electrodes(coords_array)
        
        # Choose analysis method based on electrode distribution
        if is_bilateral:
            logging.info(f"Bilateral configuration detected: {hemi_info['left_count']} left + {hemi_info['right_count']} right electrodes")
            logging.info("Using bilateral hemisphere analysis...")
            
            # Use bilateral analysis
            trajectory_results = analyze_both_hemispheres_separately(
                coords_array=coords_array,
                entry_points=None,
                max_neighbor_distance=7.5,
                min_neighbors=3,
                expected_spacing_range=(3.0, 5.0),
                use_adaptive_clustering=False  # Can be made configurable
            )
        else:
            logging.info(f"Unified analysis for {len(coords_array)} electrodes")
            
            # Use standard analysis
            trajectory_results = integrated_trajectory_analysis(
                coords_array=coords_array,
                entry_points=None,
                max_neighbor_distance=7.5,
                min_neighbors=3,
                expected_spacing_range=(3.0, 5.0)
            )
        
        if trajectory_results.get('n_trajectories', 0) == 0:
            return {
                'success': False,
                'reason': 'No trajectories detected by clustering algorithm',
                'analysis_results': trajectory_results,
                'total_points': num_points,
                'bilateral_info': hemi_info
            }
        
        # Apply hemisphere splitting if both hemispheres and cross-hemisphere trajectories exist
        if is_bilateral:
            trajectory_results = apply_hemisphere_splitting_to_results(
                trajectory_results, coords_array, hemisphere='both'
            )
        
        # Filter by trajectory IDs if specified
        if trajectory_ids is not None:
            original_trajectories = trajectory_results['trajectories']
            filtered_trajectories = []
            
            for traj in original_trajectories:
                if traj['cluster_id'] in trajectory_ids:
                    filtered_trajectories.append(traj)
            
            trajectory_results['trajectories'] = filtered_trajectories
            trajectory_results['n_trajectories'] = len(filtered_trajectories)
            
            logging.info(f"Filtered to {len(filtered_trajectories)} trajectories with IDs: {trajectory_ids}")
        
        if len(trajectory_results['trajectories']) == 0:
            return {
                'success': False,
                'reason': f'No trajectories found with specified IDs: {trajectory_ids}',
                'available_ids': [t['cluster_id'] for t in original_trajectories],
                'analysis_results': trajectory_results,
                'bilateral_info': hemi_info
            }
        
        # Calculate trajectory scores
        scores_df = calculate_trajectory_scores(
            trajectory_results['trajectories'], 
            coords_array, 
            trajectory_results
        )
        
        # Save results if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save trajectory analysis
            trajectory_csv = os.path.join(output_dir, f'trajectory_analysis_{volume_name}.csv')
            scores_df.to_csv(trajectory_csv, index=False)
            
            # Create visualization
            viz_path = os.path.join(output_dir, f'trajectory_visualization_{volume_name}.png')
            create_basic_3d_plot(coords_array, trajectory_results, scores_df, viz_path)
            
            # Create HTML report
            html_path = os.path.join(output_dir, f'trajectory_report_{volume_name}.html')
            create_interactive_annotation_report(scores_df, viz_path, html_path)
            
            logging.info(f"Results saved to: {output_dir}")
        
        logging.info("✅ Trajectory analysis completed successfully!")
        
        return {
            'success': True,
            'n_trajectories': trajectory_results['n_trajectories'],
            'trajectories': trajectory_results['trajectories'],
            'scores_df': scores_df,
            'analysis_results': trajectory_results,
            'total_points': num_points,
            'trajectory_ids_analyzed': trajectory_ids,
            'output_dir': output_dir,
            'bilateral_info': hemi_info,
            'analysis_method': 'bilateral' if is_bilateral else 'unified'
        }
        
    except Exception as e:
        logging.error(f"❌ Trajectory analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_enhanced_trajectory_analysis_with_adaptive_clustering(markup_node, trajectory_ids=None, output_dir=None, volume_name="analysis"):
    """
    Enhanced trajectory analysis with adaptive clustering and bilateral support.
    
    This function automatically determines optimal clustering parameters and uses
    bilateral analysis when appropriate.
    """
    if not TRAJECTORY_ANALYSIS_AVAILABLE:
        raise ImportError("Trajectory analysis module not available")
    
    try:
        logging.info("🔧 Starting enhanced trajectory analysis with adaptive clustering...")
        
        # Extract coordinates
        coords_list = []
        num_points = markup_node.GetNumberOfControlPoints()
        
        if num_points < 6:
            return {
                'success': False,
                'reason': f'Insufficient points ({num_points}). Need at least 6 for trajectory analysis.',
                'total_points': num_points
            }
        
        for i in range(num_points):
            pos = [0, 0, 0]
            markup_node.GetNthControlPointPosition(i, pos)
            coords_list.append(pos)
        
        coords_array = np.array(coords_list)
        logging.info(f"Extracted {len(coords_array)} electrode coordinates from markup")
        
        # Check for bilateral configuration
        is_bilateral, left_coords, right_coords, hemi_info = detect_and_split_bilateral_electrodes(coords_array)
        
        # Expected contact counts for adaptive clustering
        expected_contact_counts = [5, 8, 10, 12, 15, 18]
        
        if is_bilateral:
            logging.info(f"Bilateral configuration detected: using hemisphere-separated adaptive analysis...")
            
            # Use bilateral analysis with adaptive clustering
            trajectory_results = analyze_both_hemispheres_separately(
                coords_array=coords_array,
                entry_points=None,
                max_neighbor_distance=8,
                min_neighbors=3,
                expected_spacing_range=(3.0, 5.0),
                use_adaptive_clustering=True,
                expected_contact_counts=expected_contact_counts
            )
        else:
            logging.info("Using adaptive clustering for unified analysis...")
            
            # Find optimal parameters
            parameter_search = adaptive_clustering_parameters(
                coords_array=coords_array,
                initial_eps=8,
                initial_min_neighbors=3,
                expected_contact_counts=expected_contact_counts,
                max_iterations=10,
                eps_step=0.5,
                verbose=True
            )
            
            optimal_eps = parameter_search['optimal_eps']
            optimal_min_neighbors = parameter_search['optimal_min_neighbors']
            
            logging.info(f"Optimal parameters: eps={optimal_eps}, min_neighbors={optimal_min_neighbors}")
            
            # Run analysis with optimal parameters
            trajectory_results = integrated_trajectory_analysis(
                coords_array=coords_array,
                entry_points=None,
                max_neighbor_distance=optimal_eps,
                min_neighbors=optimal_min_neighbors,
                expected_spacing_range=(3.0, 5.0)
            )
            
            trajectory_results['parameter_search'] = parameter_search
        
        if trajectory_results.get('n_trajectories', 0) == 0:
            return {
                'success': False,
                'reason': 'No trajectories detected by adaptive clustering',
                'analysis_results': trajectory_results,
                'total_points': num_points,
                'bilateral_info': hemi_info
            }
        
        # Apply hemisphere splitting if needed
        if is_bilateral:
            trajectory_results = apply_hemisphere_splitting_to_results(
                trajectory_results, coords_array, hemisphere='both'
            )
        
        # Filter by trajectory IDs if specified
        if trajectory_ids is not None:
            original_trajectories = trajectory_results['trajectories']
            filtered_trajectories = [traj for traj in original_trajectories if traj['cluster_id'] in trajectory_ids]
            
            trajectory_results['trajectories'] = filtered_trajectories
            trajectory_results['n_trajectories'] = len(filtered_trajectories)
            
            logging.info(f"Filtered to {len(filtered_trajectories)} trajectories with IDs: {trajectory_ids}")
        
        # Calculate scores and save results (same as before)
        scores_df = calculate_trajectory_scores(
            trajectory_results['trajectories'], 
            coords_array, 
            trajectory_results
        )
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            trajectory_csv = os.path.join(output_dir, f'adaptive_trajectory_analysis_{volume_name}.csv')
            scores_df.to_csv(trajectory_csv, index=False)
            
            viz_path = os.path.join(output_dir, f'adaptive_trajectory_visualization_{volume_name}.png')
            create_basic_3d_plot(coords_array, trajectory_results, scores_df, viz_path)
            
            html_path = os.path.join(output_dir, f'adaptive_trajectory_report_{volume_name}.html')
            create_interactive_annotation_report(scores_df, viz_path, html_path)
            
            logging.info(f"Enhanced results saved to: {output_dir}")
        
        logging.info("✅ Enhanced trajectory analysis completed successfully!")
        
        return {
            'success': True,
            'n_trajectories': trajectory_results['n_trajectories'],
            'trajectories': trajectory_results['trajectories'],
            'scores_df': scores_df,
            'analysis_results': trajectory_results,
            'total_points': num_points,
            'trajectory_ids_analyzed': trajectory_ids,
            'output_dir': output_dir,
            'bilateral_info': hemi_info,
            'analysis_method': 'bilateral_adaptive' if is_bilateral else 'unified_adaptive',
            'optimization_used': True
        }
        
    except Exception as e:
        logging.error(f"❌ Enhanced trajectory analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def create_trajectory_lines_in_slicer(trajectory_results, markup_node, coords_array):
    """
    Create trajectory lines in 3D Slicer based on analysis results.
    
    Args:
        trajectory_results: Results from trajectory analysis
        markup_node: Original markup node with points
        coords_array: Array of coordinates
    
    Returns:
        list: Created trajectory line nodes
    """
    try:
        line_nodes = []
        
        if 'graph' not in trajectory_results:
            logging.warning("No graph data in trajectory results")
            return line_nodes
        
        clusters = np.array([node[1]['dbscan_cluster'] for node in trajectory_results['graph'].nodes(data=True)])
        
        for trajectory in trajectory_results['trajectories']:
            cluster_id = trajectory['cluster_id']
            
            # Find points belonging to this trajectory
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            # Get coordinates and sort along trajectory
            cluster_coords = coords_array[cluster_indices]
            
            if 'direction' in trajectory and len(cluster_coords) > 2:
                direction = np.array(trajectory['direction'])
                center = np.mean(cluster_coords, axis=0)
                projected = np.dot(cluster_coords - center, direction)
                sorted_indices = np.argsort(projected)
                cluster_coords = cluster_coords[sorted_indices]
            
            # Create line markup node
            line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", 
                                                          f"Trajectory_{cluster_id}")
            
            # Add points to create trajectory line
            for i, coord in enumerate(cluster_coords):
                if i == 0:
                    line_node.SetNthControlPointPosition(0, coord[0], coord[1], coord[2])
                elif i == 1:
                    line_node.SetNthControlPointPosition(1, coord[0], coord[1], coord[2])
                else:
                    # For additional points, we need to create curve nodes
                    break
            
            # Set line properties
            display_node = line_node.GetDisplayNode()
            if display_node:
                # Color based on trajectory quality
                n_contacts = trajectory.get('electrode_count', 0)
                if n_contacts >= 8:
                    display_node.SetColor(0.0, 1.0, 0.0)  # Green for good trajectories
                elif n_contacts >= 5:
                    display_node.SetColor(1.0, 0.5, 0.0)  # Orange for medium
                else:
                    display_node.SetColor(1.0, 0.0, 0.0)  # Red for poor
                
                display_node.SetLineThickness(3.0)
                display_node.SetOpacity(0.8)
            
            line_nodes.append(line_node)
            logging.info(f"Created trajectory line for cluster {cluster_id} with {len(cluster_coords)} points")
        
        return line_nodes
        
    except Exception as e:
        logging.error(f"Error creating trajectory lines: {e}")
        return []


#
# SEEG_masking
#

class SEEG_ElectrodeLocalization(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SEEG Electrode Localization")
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Rocio Avalos (UPC)"]
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SEEG_masking">module documentation</a>.
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # SEEG_masking1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SEEG_masking",
        sampleName="SEEG_masking1",
        thumbnailFileName=os.path.join(iconsPath, "SEEG_masking1.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SEEG_masking1.nrrd",
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        nodeNames="SEEG_masking1",
    )

    # SEEG_masking2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SEEG_masking",
        sampleName="SEEG_masking2",
        thumbnailFileName=os.path.join(iconsPath, "SEEG_masking2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SEEG_masking2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        nodeNames="SEEG_masking2",
    )



#############
## Confidence widget
######################
class ConfidenceThresholdWidget:
    """Handler for confidence viewer UI elements created in Qt Designer."""
    
    def __init__(self, ui_elements):
        """Initialize with UI elements from Qt Designer."""
        self.ui = ui_elements
        self.csv_path = None
        self.points = []
        self.threshold = 0.05
        self.markupNode = None
        self.setupConnections()
        
    def setupConnections(self):
        """Connect Qt Designer UI elements to functionality."""
        # Connect slider to threshold update
        if hasattr(self.ui, 'confidenceThresholdSlider'):
            self.ui.confidenceThresholdSlider.valueChanged.connect(self.updateThreshold)
        
        # Connect buttons if they exist
        if hasattr(self.ui, 'refreshPointsButton'):
            self.ui.refreshPointsButton.clicked.connect(self.refreshPoints)
        if hasattr(self.ui, 'clearPointsButton'):
            self.ui.clearPointsButton.clicked.connect(self.clearPoints)
    
    def loadCsvFile(self, csv_path):
        """Load electrode data from confidence predictions CSV file."""
        try:
            self.csv_path = csv_path
            self.points = []
            
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        x = float(row['RAS_X'])
                        y = float(row['RAS_Y'])
                        z = float(row['RAS_Z'])
                        confidence = float(row['Ensemble_Confidence'])
                        self.points.append((x, y, z, confidence))
                    except (KeyError, ValueError) as e:
                        print(f"Skipping row due to error: {e}")
            
            # Update status label
            filename = os.path.basename(csv_path)
            if hasattr(self.ui, 'csvPathLabel'):
                self.ui.csvPathLabel.setText(f"📊 Loaded: {filename}")
                self.ui.csvPathLabel.setStyleSheet("color: #28a745; font-weight: bold;")
            
            # Enable buttons if they exist
            if hasattr(self.ui, 'refreshPointsButton'):
                self.ui.refreshPointsButton.setEnabled(True)
            if hasattr(self.ui, 'clearPointsButton'):
                self.ui.clearPointsButton.setEnabled(True)
            
            # Auto-refresh display
            self.refreshPoints()
            
            print(f"✅ Loaded {len(self.points)} electrode points from {filename}")
            
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            if hasattr(self.ui, 'csvPathLabel'):
                self.ui.csvPathLabel.setText(f"❌ Error loading: {os.path.basename(csv_path)}")
                self.ui.csvPathLabel.setStyleSheet("color: #dc3545;")
    
    def updateThreshold(self, value):
        """Update confidence threshold and refresh display."""
        self.threshold = value / 100.0
        
        # Update threshold label
        if hasattr(self.ui, 'thresholdValueLabel'):
            self.ui.thresholdValueLabel.setText(f"Threshold: {value}")
        
        # Refresh points if we have data
        if self.points:
            self.refreshPoints()
    
    def refreshPoints(self):
        """Refresh the displayed electrode points based on current threshold."""
        if not self.points:
            return
            
        # Create or get existing markup node
        if not self.markupNode:
            self.markupNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", 
                "Electrode_Predictions"
            )
            # Set display properties
            displayNode = self.markupNode.GetDisplayNode()
            if displayNode:
                displayNode.SetGlyphScale(2.5) 
                displayNode.SetColor(1.0, 0.5, 0.0)  
                displayNode.SetTextScale(0)  # Hide all text labels
        else:
            self.markupNode.RemoveAllControlPoints()
        
        # Add points that meet threshold
        visible_count = 0
        for x, y, z, confidence in self.points:
            if confidence >= self.threshold:
                point_index = self.markupNode.AddControlPoint([x, y, z])
                visible_count += 1
        
        # Update statistics
        self.updateStatistics()
        
        print(f"📊 Displaying {visible_count}/{len(self.points)} electrodes above threshold {self.threshold:.2f}")
    
    def updateStatistics(self):
        """Update the statistics display."""
        if not self.points:
            return
            
        total = len(self.points)
        visible = sum(1 for _, _, _, conf in self.points if conf >= self.threshold)
        high_conf = sum(1 for _, _, _, conf in self.points if conf >= 0.8)
        medium_conf = sum(1 for _, _, _, conf in self.points if 0.6 <= conf < 0.8)
        
        # Update labels if they exist
        if hasattr(self.ui, 'totalElectrodesLabel'):
            self.ui.totalElectrodesLabel.setText(f"Total electrodes: {total}")
        if hasattr(self.ui, 'visibleElectrodesLabel'):
            self.ui.visibleElectrodesLabel.setText(f"Visible electrodes: {visible}")
        if hasattr(self.ui, 'highConfidenceLabel'):
            self.ui.highConfidenceLabel.setText(f"High confidence (≥0.8): {high_conf}")
        if hasattr(self.ui, 'mediumConfidenceLabel'):
            self.ui.mediumConfidenceLabel.setText(f"Medium confidence (0.6-0.8): {medium_conf}")
        
        # Update simple stats label if that's what you have
        if hasattr(self.ui, 'quickStatsLabel'):
            self.ui.quickStatsLabel.setText(f"{visible} electrodes visible")
    
    def clearPoints(self):
        """Clear all displayed electrode points."""
        if self.markupNode:
            slicer.mrmlScene.RemoveNode(self.markupNode)
            self.markupNode = None
        print("🧹 Cleared all electrode points")

### TRAJECTORY ANALYSIS FUNCTIONS ###
    def parseTrajectoryIds(self):
        """Parse trajectory IDs from the input field."""
        trajectory_ids = None
        if hasattr(self.ui, 'trajectoryIdsLineEdit'):
            ids_text = self.ui.trajectoryIdsLineEdit.text.strip()
            if ids_text:
                try:
                    trajectory_ids = [int(x.strip()) for x in ids_text.split(',')]
                    print(f"Parsed trajectory IDs: {trajectory_ids}")
                except ValueError:
                    slicer.util.warningDisplay("Invalid trajectory IDs format. Please use format: 0,1,2")
                    return None
        return trajectory_ids

    def createBlueTrajectoryLines(self, trajectory_results, markup_node, coords_array, requested_ids):
        """
        Create blue trajectory lines in 3D Slicer for specified trajectory IDs.
        
        Args:
            trajectory_results: Results from trajectory analysis
            markup_node: Original markup node with points
            coords_array: Array of coordinates
            requested_ids: List of trajectory IDs to create lines for
        
        Returns:
            list: Created trajectory line nodes
        """
        try:
            line_nodes = []
            
            if 'graph' not in trajectory_results:
                logging.warning("No graph data in trajectory results")
                return line_nodes
            
            # Get cluster assignments for each point
            clusters = np.array([node[1]['dbscan_cluster'] for node in trajectory_results['graph'].nodes(data=True)])
            
            # Create lines only for requested trajectory IDs
            for trajectory in trajectory_results['trajectories']:
                cluster_id = trajectory['cluster_id']
                
                # Skip if this trajectory ID was not requested
                if cluster_id not in requested_ids:
                    continue
                
                # Find points belonging to this trajectory
                cluster_mask = clusters == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) < 2:
                    logging.warning(f"Trajectory {cluster_id} has less than 2 points, skipping line creation")
                    continue
                
                # Get coordinates and sort along trajectory direction
                cluster_coords = coords_array[cluster_indices]
                
                # Sort points along the trajectory direction if available
                if 'direction' in trajectory and len(cluster_coords) > 2:
                    direction = np.array(trajectory['direction'])
                    center = np.mean(cluster_coords, axis=0)
                    projected = np.dot(cluster_coords - center, direction)
                    sorted_indices = np.argsort(projected)
                    cluster_coords = cluster_coords[sorted_indices]
                
                # Create curve markup node for complex trajectories
                if len(cluster_coords) > 2:
                    curve_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", 
                                                                f"Trajectory_Line_{cluster_id}")
                    
                    # Add all points to the curve
                    for i, coord in enumerate(cluster_coords):
                        curve_node.AddControlPoint(coord[0], coord[1], coord[2])
                    
                    # Set curve properties
                    curve_node.SetCurveTypeToLinear()  # Linear interpolation between points
                    
                    # Set blue color
                    display_node = curve_node.GetDisplayNode()
                    if display_node:
                        display_node.SetColor(0.0, 0.0, 1.0)  # Blue color (RGB)
                        display_node.SetLineThickness(3.0)
                        display_node.SetOpacity(1.0)
                        display_node.SetTextScale(0)  # Hide labels
                    
                    line_nodes.append(curve_node)
                    logging.info(f"Created blue trajectory curve for cluster {cluster_id} with {len(cluster_coords)} points")
                
                else:
                    # For 2-point trajectories, use line markup
                    line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", 
                                                                f"Trajectory_Line_{cluster_id}")
                    
                    # Add start and end points
                    line_node.SetNthControlPointPosition(0, cluster_coords[0][0], cluster_coords[0][1], cluster_coords[0][2])
                    line_node.SetNthControlPointPosition(1, cluster_coords[1][0], cluster_coords[1][1], cluster_coords[1][2])
                    
                    # Set blue color
                    display_node = line_node.GetDisplayNode()
                    if display_node:
                        display_node.SetColor(0.0, 0.0, 1.0)  # Blue color (RGB)
                        display_node.SetLineThickness(3.0)
                        display_node.SetOpacity(1.0)
                        display_node.SetTextScale(0)  # Hide labels
                    
                    line_nodes.append(line_node)
                    logging.info(f"Created blue trajectory line for cluster {cluster_id} with 2 points")
            
            return line_nodes
            
        except Exception as e:
            logging.error(f"Error creating blue trajectory lines: {e}")
            return []

#
# SEEG_maskingWidget - Pure Direct UI Access
#

class SEEG_ElectrodeLocalizationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class with pure direct UI access."""

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.outputVolumeNode = None  # Track the generated mask volume
        self.confidenceWidget = None # Confidence viewer widget


    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SEEG_ElectrodeLocalization.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class
        self.logic = SEEG_ElectrodeLocalizationLogic()

        # Connect Qt Designer widgets
        self.ui.showPathButton.connect("clicked(bool)", self.onBrowseOutput)
        self.ui.folderNameLineEdit.setText("")

        # Debug and configure existing elements
        self.debugUIElements()
        self.configureVolumeSelectors()

        # Connect buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        #self.ui.huggingFaceLinkButton.connect("clicked(bool)", self.onOpenHuggingFaceLink)

        # Trajectory analysis setup
        self.setupTrajectoryAnalysis()
        

        # Connect volume selectors
        if hasattr(self.ui, 'inputSelector'):
            self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        if hasattr(self.ui, 'inputSelectorCT'):
            self.ui.inputSelectorCT.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)

        # === BRAIN MASK OPTION INITIALIZATION ===
        self.setupBrainMaskOption()

        # === CONFIDENCE VIEWER INITIALIZATION ===
        self.setupConfidenceViewer()

        # Set default output path
        self.setDefaultOutputPath()
        
        # Initial button state check
        self.updateApplyButtonState()

    def debugUIElements(self):
        """Debug what UI elements we have access to."""
        print("=== UI Elements Debug ===")
        print(f"Available UI attributes: {[attr for attr in dir(self.ui) if not attr.startswith('_')]}")
        print(f"Has inputSelector: {hasattr(self.ui, 'inputSelector')}")
        print(f"Has inputSelectorCT: {hasattr(self.ui, 'inputSelectorCT')}")
        
        if hasattr(self.ui, 'inputSelector'):
            print(f"inputSelector type: {type(self.ui.inputSelector)}")
        if hasattr(self.ui, 'inputSelectorCT'):
            print(f"inputSelectorCT type: {type(self.ui.inputSelectorCT)}")

    def configureVolumeSelectors(self):
        """Configure the volume selectors explicitly."""
        print("=== Configuring Volume Selectors ===")
        
        # Configure MRI input selector
        if hasattr(self.ui, 'inputSelector'):
            self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)
            self.ui.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            self.ui.inputSelector.selectNodeUponCreation = False
            self.ui.inputSelector.addEnabled = False
            self.ui.inputSelector.removeEnabled = False
            self.ui.inputSelector.noneEnabled = True
            self.ui.inputSelector.showHidden = False
            self.ui.inputSelector.showChildNodeTypes = False
            self.ui.inputSelector.setToolTip("Select MRI volume for brain mask generation")
            print("✓ inputSelector configured")
        else:
            print("✗ inputSelector not found!")
        
        # Configure CT input selector
        if hasattr(self.ui, 'inputSelectorCT'):
            self.ui.inputSelectorCT.setMRMLScene(slicer.mrmlScene)
            self.ui.inputSelectorCT.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            self.ui.inputSelectorCT.selectNodeUponCreation = False
            self.ui.inputSelectorCT.addEnabled = False
            self.ui.inputSelectorCT.removeEnabled = False
            self.ui.inputSelectorCT.noneEnabled = True
            self.ui.inputSelectorCT.showHidden = False
            self.ui.inputSelectorCT.showChildNodeTypes = False
            self.ui.inputSelectorCT.setToolTip("Select CT volume for enhancement processing")
            print("✓ inputSelectorCT configured")
        else:
            print("✗ inputSelectorCT not found!")

        # Check available volumes
        volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        print(f"Available volumes in scene: {len(volumes)}")
        for i, vol in enumerate(volumes):
            print(f"  {i+1}: {vol.GetName()}")

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Refresh selectors when entering
        self.configureVolumeSelectors()
        self.updateApplyButtonState()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        pass

    def onInputChanged(self) -> None:
        """Called when input volume selection changes."""
        print("Input selection changed")
        self.updateApplyButtonState()

    def setupBrainMaskOption(self):
        """Setup the brain mask generation link."""
        try:
            # Connect the Hugging Face link button
            if hasattr(self.ui, 'huggingFaceLinkButton'):
                self.ui.huggingFaceLinkButton.clicked.connect(self.onOpenHuggingFaceLink)
                print("✅ Brain mask link button connected")
            else:
                print("⚠️ huggingFaceLinkButton not found in UI")
                
        except Exception as e:
            print(f"⚠️ Brain mask setup failed: {e}")

    def onOpenHuggingFaceLink(self):
        """Open the Hugging Face brain mask generation link."""
        import webbrowser
        url = "https://huggingface.co/spaces/rocioavl/seeg-brain-mask-segmentation"
        
        # Show informative dialog
        msg = qt.QMessageBox()
        msg.setWindowTitle("Generate Brain Mask Online")
        msg.setText("This will open our Hugging Face 🤗 brain mask generator.")
        msg.setInformativeText(
            "Steps:\n"
            "1. Upload your post-operative CT scan\n"
            "2. Wait for processing (~2-3 minutes)\n"
            "3. Download the generated brain mask\n"
            "4. Load it as 'MRI input' in this module\n"
            "5. Check 'Input is already a binary brain mask'\n\n"
            f"URL: {url}"
        )
        msg.setStandardButtons(qt.QMessageBox.Ok | qt.QMessageBox.Cancel)
        msg.setDefaultButton(qt.QMessageBox.Ok)
        
        if msg.exec_() == qt.QMessageBox.Ok:
            try:
                webbrowser.open(url)
                print(f"✅ Opened Hugging Face brain mask generator: {url}")
            except Exception as e:
                print(f"❌ Failed to open browser: {e}")
                # Fallback - copy to clipboard
                clipboard = qt.QApplication.clipboard()
                clipboard.setText(url)
                slicer.util.infoDisplay(
                    f"Could not open browser automatically.\n\n"
                    f"Link copied to clipboard:\n{url}\n\n"
                    f"Paste this into your browser to access the brain mask generator."
                )

    def setupTrajectoryAnalysis(self):
        """Setup the trajectory analysis section matching the Qt Designer UI."""
        try:
            # Configure markup selector for electrode markups
            if hasattr(self.ui, 'markupSelector'):
                self.ui.markupSelector.setMRMLScene(slicer.mrmlScene)
                self.ui.markupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
                self.ui.markupSelector.selectNodeUponCreation = False
                self.ui.markupSelector.addEnabled = False
                self.ui.markupSelector.removeEnabled = False
                self.ui.markupSelector.noneEnabled = True
                self.ui.markupSelector.showHidden = False
                self.ui.markupSelector.showChildNodeTypes = False
                self.ui.markupSelector.setToolTip("Select markup node containing electrode points")
                print("✅ Markup selector configured")
            
            # Connect "Generate Reports and CSV" button
            if hasattr(self.ui, 'generateReportsAndCSVButton'):
                self.ui.generateReportsAndCSVButton.clicked.connect(self.onGenerateReportsAndCSV)
                print("✅ Generate Reports and CSV button connected")
            
            # Connect "Create Trajectory Lines" button  
            if hasattr(self.ui, 'createTrajectoryLinesButton'):
                self.ui.createTrajectoryLinesButton.clicked.connect(self.onCreateTrajectoryLines)
                print("✅ Create Trajectory Lines button connected")
                
            # Setup trajectory IDs input field
            if hasattr(self.ui, 'trajectoryIdsLineEdit'):
                self.ui.trajectoryIdsLineEdit.setPlaceholderText("comma-separated, e.g., 0,1,2")
                print("✅ Trajectory IDs input configured")
                
            # Initially disable buttons until markup is selected
            self.updateTrajectoryButtonStates()
            
            # Connect markup selector to update button states
            if hasattr(self.ui, 'markupSelector'):
                self.ui.markupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateTrajectoryButtonStates)
                
        except Exception as e:
            print(f"⚠️ Trajectory analysis setup failed: {e}")


    def getSelectedMarkupNode(self):
        """Get the currently selected markup node."""
        if hasattr(self.ui, 'markupSelector'):
            return self.ui.markupSelector.currentNode()
        return None 
    
    def updateTrajectoryButtonStates(self):
        """Enable/disable trajectory buttons based on markup selection."""
        markup_node = self.getSelectedMarkupNode()
        has_markup = markup_node is not None
        
        if hasattr(self.ui, 'generateReportsAndCSVButton'):
            self.ui.generateReportsAndCSVButton.setEnabled(has_markup)
        if hasattr(self.ui, 'createTrajectoryLinesButton'):
            self.ui.createTrajectoryLinesButton.setEnabled(has_markup)

######## Output path handling ########
    def onBrowseOutput(self) -> None:
        """Handle browse button click - shows current folder structure."""
        current_folder_name = self.ui.folderNameLineEdit.text.strip()
        
        if not current_folder_name:
            from datetime import datetime
            current_date = datetime.now()
            current_folder_name = f"results_{current_date.strftime('%d_%m_%y')}"
        
        # Show where it will be saved
        full_path = os.path.join(os.path.expanduser("~"), "Documents", "SEEG_Results", current_folder_name)
        
        msg = qt.QMessageBox()
        msg.setWindowTitle("Output Location")
        msg.setText(f"Results will be saved to:\n\n{full_path}")
        msg.setInformativeText("This location is fixed. You can only change the folder name above.")
        msg.setStandardButtons(qt.QMessageBox.Ok)
        msg.exec_()

    def setDefaultOutputPath(self) -> None:
        """Set a default folder name suggestion."""
        # Leave empty so it shows the placeholder text
        self.ui.folderNameLineEdit.setText("")

    def getOutputDirectory(self) -> str:
        """Get the output directory with user folder name or timestamped fallback."""
        # Always base path: Documents/SEEG_Results/
        base_path = os.path.join(os.path.expanduser("~"), "Documents", "SEEG_Results")
        
        # Get user folder name
        user_folder_name = self.ui.folderNameLineEdit.text.strip()
        
        if user_folder_name:
            # User specified a folder name - clean it up for filesystem
            # Remove invalid characters and spaces
            import re
            clean_folder_name = re.sub(r'[<>:"/\\|?*]', '_', user_folder_name)
            clean_folder_name = clean_folder_name.strip()
            folder_name = clean_folder_name
        else:
            # No user input - use timestamped fallback
            from datetime import datetime
            current_date = datetime.now()
            folder_name = f"results_{current_date.strftime('%d_%m_%y')}"
        
        # Final directory: Documents/SEEG_Results/[folder_name]
        main_output_dir = os.path.join(base_path, folder_name)
        
        logging.info(f"Using output directory: {main_output_dir}")
        return main_output_dir


    def updateApplyButtonState(self) -> None:
        """Update the Apply button state based on current selections."""
        # Get current selections
        input_volume = None
        input_volume_CT = None
        
        if hasattr(self.ui, 'inputSelector'):
            input_volume = self.ui.inputSelector.currentNode()
        if hasattr(self.ui, 'inputSelectorCT'):
            input_volume_CT = self.ui.inputSelectorCT.currentNode()

        print(f"Current selections - MRI: {input_volume.GetName() if input_volume else 'None'}, CT: {input_volume_CT.GetName() if input_volume_CT else 'None'}")

        # Determine button state
        canApply = (input_volume is not None and input_volume_CT is not None)
        
        if not input_volume and not input_volume_CT:
            tooltip = "Select both MRI and CT input volumes"
        elif not input_volume:
            tooltip = "Select MRI input volume for brain mask generation"
        elif not input_volume_CT:
            tooltip = "Select CT input volume for enhancement processing"
        else:
            tooltip = "Generate brain mask and enhance CT volume"

        # Update button
        if hasattr(self.ui, 'applyButton'):
            self.ui.applyButton.enabled = canApply
            self.ui.applyButton.toolTip = _(tooltip)


    def setupConfidenceViewer(self):
        try:
            self.confidenceWidget = ConfidenceThresholdWidget(self.ui)  
            print("✅ Confidence viewer initialized successfully")
        except Exception as e:
            print(f"⚠️ Confidence viewer setup failed: {e}")
            self.confidenceWidget = None


    def onApplyButton(self) -> None:
        """Run processing when user clicks 'Apply' button - no directory dialogs!"""
        print("Apply button clicked!")
        
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            
            # Get input volumes
            input_volume_node = None
            input_volume_node_CT = None
            
            if hasattr(self.ui, 'inputSelector'):
                input_volume_node = self.ui.inputSelector.currentNode()
            if hasattr(self.ui, 'inputSelectorCT'):
                input_volume_node_CT = self.ui.inputSelectorCT.currentNode()
            # CHECK IF BINARY MASK OPTION IS SELECTED
            is_binary_mask = hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked()
            
            # Validate inputs
            if not input_volume_node:
                slicer.util.errorDisplay("Please select an MRI input volume for brain mask generation.")
                return
                
            if not input_volume_node_CT:
                slicer.util.errorDisplay("Please select a CT input volume for enhancement processing.")
                return
            
            logging.info(f"Processing MRI volume: {input_volume_node.GetName()}")
            logging.info(f"Processing CT volume: {input_volume_node_CT.GetName()}")
            
            # === GET OUTPUT DIRECTORY (NO DIALOGS!) ===
            main_output_dir = self.getOutputDirectory()
            
            # Create organized subfolders
            brain_mask_dir = os.path.join(main_output_dir, "Brain_mask")
            enhanced_masks_dir = os.path.join(main_output_dir, "Enhanced_masks")
            
            try:
                os.makedirs(brain_mask_dir, exist_ok=True)
                os.makedirs(enhanced_masks_dir, exist_ok=True)
                logging.info(f"Created output directories in: {main_output_dir}")
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to create output directories: {str(e)}")
                return
            
            # === GENERATE BRAIN MASK ===
            try:
                if input_volume_node and is_binary_mask:
                    # Use existing binary mask directly
                    logging.info(f"Using existing binary mask: {input_volume_node.GetName()}")
                    self.outputVolumeNode = input_volume_node  
                    
                elif input_volume_node:
                    # Generate brain mask from MRI
                    logging.info(f"Processing MRI volume: {input_volume_node.GetName()}")
                    self.outputVolumeNode = self.logic.process(input_volume_node)
                    
                else:
                    # No MRI provided - this is now required
                    slicer.util.errorDisplay("Please select an MRI input volume for brain mask generation.")
                    return
                
                if not self.outputVolumeNode:
                    slicer.util.errorDisplay("Failed to generate brain mask.")
                    return
                
                # Auto-save brain mask
                brain_mask_filename = f"BrainMask_{input_volume_node.GetName()}.nrrd"
                brain_mask_path = os.path.join(brain_mask_dir, brain_mask_filename)
                
                success = slicer.util.saveNode(self.outputVolumeNode, brain_mask_path)
                if success:
                    logging.info(f"Brain mask saved to: {brain_mask_path}")
                
            except Exception as e:
                slicer.util.errorDisplay(f"Error generating brain mask: {str(e)}")
                return
            
            # === GENERATE ENHANCED CT VOLUMES ===
            try:
                from Threshold_mask.ctp_enhancer import CTPEnhancer
                
                ctp_enhancer = CTPEnhancer()
                
                enhanced_volumes = ctp_enhancer.enhance_ctp(
                    inputVolume=input_volume_node_CT,
                    inputROI=self.outputVolumeNode,
                    outputDir=enhanced_masks_dir,
                    model_path=MODEL_PATH,
                )
                
                if enhanced_volumes and len(enhanced_volumes) > 0:
                    volume_names = list(enhanced_volumes.keys())
                    logging.info(f"Enhanced CT volumes generated: {', '.join(volume_names)}")
                    logging.info(f"All results saved to: {main_output_dir}")
                else:
                    slicer.util.warningDisplay("No enhanced volumes were generated.")
                    
            except Exception as e:
                slicer.util.errorDisplay(f"Error during CT enhancement: {str(e)}")
                logging.error(f"CT enhancement error: {str(e)}")

            # === GENERATE GLOBAL MASKS ===
            try:
                # Import the EnhancedMaskSelector
                from masks_fusion_top import EnhancedMaskSelector
                
                # Create Global_masks subdirectory
                global_masks_dir = os.path.join(main_output_dir, "Global_masks")
                os.makedirs(global_masks_dir, exist_ok=True)
                
                # ⭐  filename with CT volume name ⭐
                excluded_file = f"Filtered_DESCARGAR_roi_volume_features_{input_volume_node_CT.GetName()}.nrrd"
                confidence_ct_source = os.path.join(enhanced_masks_dir, excluded_file)
                confidence_ct_backup = os.path.join(main_output_dir, "confidence_ct_file.nrrd")

                
                if os.path.exists(confidence_ct_source):
                    import shutil
                    shutil.copy2(confidence_ct_source, confidence_ct_backup)
                    logging.info(f"✅ Saved confidence CT file: {confidence_ct_backup}")
                else:
                    confidence_ct_backup = None
                    logging.warning(f"⚠️ Confidence CT file not found: {excluded_file}")
                
                # Now proceed with global masking (excluding the file as before)
                enhanced_mask_files = []
                for file in os.listdir(enhanced_masks_dir):
                    if file.endswith('.nrrd') and file != excluded_file:
                        enhanced_mask_files.append(file)
                
                if len(enhanced_mask_files) < 4:
                    slicer.util.warningDisplay(f"Only {len(enhanced_mask_files)} valid enhanced masks found. Need at least 4 for global masking.")
                    global_mask_info = ""
                else:
                    logging.info(f"Found {len(enhanced_mask_files)} valid enhanced masks for global processing")
                    
                    # Initialize the enhanced mask selector (using current constructor)
                    selector = EnhancedMaskSelector(enhanced_masks_dir, global_masks_dir)
                    
                    # Manually remove the excluded file from loaded masks if it exists
                    excluded_mask_key = excluded_file.replace('.nrrd', '')
                    if excluded_mask_key in selector.masks:
                        del selector.masks[excluded_mask_key]
                        print(f"Removed excluded file from processing: {excluded_file}")
                    
                    # Select the best 4 masks
                    selected_masks = selector.select_best_masks(n_masks=4)
                    

                    if len(selected_masks) >= 2:
                        # Save top 2 individual masks (KEEP THIS)
                        for i, mask_name in enumerate(selected_masks[:2], 1):
                            output_name = f"top_mask_{i}_{input_volume_node_CT.GetName()}"
                            selector.save_mask(selector.masks[mask_name], output_name)
                        
                        # === CONSENSUS MASK ===
                        # Create 50% consensus mask instead of progressive masks
                        num_masks = len(selector.masks)
                        consensus_threshold = max(1, int(0.5 * num_masks))  # At least 50% agreement
                        consensus_mask = np.where(selector.global_vote_map >= consensus_threshold, 1, 0)
                        consensus_output_name = f"consensus_50pct_{input_volume_node_CT.GetName()}"
                        selector.save_mask(consensus_mask, consensus_output_name)
                        
                        # Create voxel count comparison plot (KEEP THIS)
                        try:
                            if hasattr(selector, 'plot_voxel_count_comparison'):
                                plots_dir = os.path.join(global_masks_dir, "plots")
                                selector.plot_voxel_count_comparison(selected_masks, plots_dir)
                                plot_info = "\n    • Voxel count comparison plot"
                            else:
                                # Simple voxel count calculation for logging
                                all_masks = list(selector.masks.keys())
                                original_voxels = sum(np.sum(selector.masks[name]) for name in all_masks)
                                selected_voxels = sum(np.sum(selector.masks[name]) for name in selected_masks)
                                logging.info(f"Voxel reduction: {original_voxels:,} → {selected_voxels:,}")
                                plot_info = ""
                        except Exception as plot_error:
                            logging.warning(f"Failed to create voxel count plot: {plot_error}")
                            plot_info = ""
                        
                        logging.info(f"Global masking completed: 2 top masks + 1 consensus mask saved to {global_masks_dir}")
                        
                        # === UPDATE THE SUCCESS MESSAGE ===
                        global_mask_info = (
                            f"\n📄 Global masks (3): Global_masks/\n"
                            f"    • 2 top individual masks\n"
                            f"    • 1 consensus mask (≥{consensus_threshold}/{num_masks} algorithm agreement){plot_info}"
                        )
                        
                    else:
                        slicer.util.warningDisplay("Not enough quality masks found for global masking.")
                        global_mask_info = ""
                        
            except ImportError as e:
                slicer.util.errorDisplay(f"Failed to import EnhancedMaskSelector: {str(e)}")
                logging.error(f"Global masking import error: {str(e)}")
                global_mask_info = ""
            except Exception as e:
                slicer.util.errorDisplay(f"Error during global masking: {str(e)}")
                logging.error(f"Global masking error: {str(e)}")
                global_mask_info = ""


            # === GENERATE ELECTRODE CONFIDENCE ANALYSIS (EMBEDDED) ===
            confidence_info = ""
            
            try:
                logging.info("Starting embedded confidence analysis...")
                
                # Create Confidence_Analysis subdirectory
                confidence_dir = os.path.join(main_output_dir, "Confidence_Analysis")
                os.makedirs(confidence_dir, exist_ok=True)
                
                # Define required file paths
                brain_mask_file = os.path.join(brain_mask_dir, brain_mask_filename)
                top_mask_file = os.path.join(global_masks_dir, f"consensus_50pct_{input_volume_node_CT.GetName()}.nrrd")
                
                # Use the backup confidence CT file
                if 'confidence_ct_backup' in locals() and confidence_ct_backup and os.path.exists(confidence_ct_backup):
                    confidence_ct_file = confidence_ct_backup
                else:
                    confidence_ct_file = None
                
                # Check if confidence analysis is possible
                missing_files = []
                if not CONFIDENCE_AVAILABLE:
                    missing_files.append("Confidence analysis dependencies")
                if not os.path.exists(brain_mask_file):
                    missing_files.append(f"Brain mask: {brain_mask_filename}")
                if not os.path.exists(top_mask_file):
                    missing_files.append(f"Consensus mask: consensus_50pct_{input_volume_node_CT.GetName()}.nrrd")
                if not confidence_ct_file or not os.path.exists(confidence_ct_file):
                    missing_files.append("Confidence CT file")
                if not os.path.exists(CONFIDENCE_MODEL_PATH):
                    missing_files.append(f"Confidence model: {os.path.basename(CONFIDENCE_MODEL_PATH)}")
                
                if missing_files:
                    missing_files_str = "\n    • ".join(missing_files)
                    logging.warning(f"Confidence analysis skipped. Missing:\n    • {missing_files_str}")
                    confidence_info = ""
                else:
                    # Run the embedded confidence analysis
                    results = run_embedded_confidence_analysis(
                        brain_mask_file=brain_mask_file,
                        top_mask_file=top_mask_file,
                        enhanced_ct_file=confidence_ct_file,
                        volume_name=input_volume_node_CT.GetName(),
                        confidence_dir=confidence_dir,
                        model_path=CONFIDENCE_MODEL_PATH
                    )
                    
                    logging.info(f"Embedded confidence analysis completed: {results['total_electrodes']} electrodes analyzed")
                    
                    # Add confidence info to success message
                    confidence_info = (
                        f"\n📊 Confidence Analysis: Confidence_Analysis/\n"
                        f"    • {results['total_electrodes']} electrode candidates analyzed\n"
                        f"    • High confidence (≥80%): {results['high_confidence']} electrodes\n"
                        f"    • Medium confidence (60-80%): {results['medium_confidence']} electrodes\n"
                        f"    • Top confidence score: {results['top_confidence']:.3f}\n"
                        f"    • Average confidence: {results['avg_confidence']:.3f}"
                    )

                    # ✅ AUTO-LOAD THE CONFIDENCE CSV 
                    if 'predictions_csv' in results:
                        predictions_csv_path = results['predictions_csv']
                        if os.path.exists(predictions_csv_path) and self.confidenceWidget:
                            self.confidenceWidget.loadCsvFile(predictions_csv_path)
                            
                            # Expand the confidence viewer section
                            if hasattr(self.ui, 'confidenceCollapsibleButton'):
                                self.ui.confidenceCollapsibleButton.collapsed = False
                            
                            print(f"✅ Auto-loaded confidence results into viewer: {os.path.basename(predictions_csv_path)}")
                            
                            # Update the confidence info to mention the viewer
                            confidence_info += f"\n    • Interactive viewer loaded with {results['total_electrodes']} electrodes"
                        
            except Exception as e:
                logging.error(f"Error during embedded confidence analysis: {str(e)}")
                slicer.util.warningDisplay(f"Confidence analysis failed: {str(e)}")
                confidence_info = ""


            # === SINGLE FINAL SUCCESS MESSAGE ===
            if enhanced_volumes and len(enhanced_volumes) > 0:
                # Count files in each directory
                brain_mask_files = [f for f in os.listdir(brain_mask_dir) if f.endswith('.nrrd')]
                enhanced_files = [f for f in os.listdir(enhanced_masks_dir) if f.endswith('.nrrd')]
                
                # Build concise folder structure
                folder_structure = f"📁 {main_output_dir}\n"
                folder_structure += f"  ├── Brain_mask/: {len(brain_mask_files)} file\n"
                folder_structure += f"  ├── Enhanced_masks/: {len(enhanced_files)} files\n"
                
                # Add global masks info if available
                if os.path.exists(global_masks_dir):
                    global_mask_files = [f for f in os.listdir(global_masks_dir) if f.endswith('.nrrd')]
                    folder_structure += f"  ├── Global_masks/: {len(global_mask_files)} files\n"
                
                # Add confidence info if available
                if os.path.exists(confidence_dir):
                    confidence_files = [f for f in os.listdir(confidence_dir) if f.endswith('.csv') or f.endswith('.txt')]
                    folder_structure += f"  └── Confidence_Analysis/: {len(confidence_files)} files\n"
                else:
                    folder_structure = folder_structure.rstrip('\n') + '\n'
                
                # Add markups info
                markups_info = ""
                if hasattr(self, 'confidenceWidget') and self.confidenceWidget and self.confidenceWidget.markupNode:
                    num_electrodes = self.confidenceWidget.markupNode.GetNumberOfControlPoints()
                    markups_info = f"\n🎯 {num_electrodes} electrode candidates loaded in 3D Slicer"
                
                final_success_message = f"✅ Processing completed successfully!\n\n{folder_structure}{markups_info}"
                
                slicer.util.infoDisplay(final_success_message)
                logging.info(f"All results saved to: {main_output_dir}")
            else:
                slicer.util.errorDisplay("Processing failed - no enhanced volumes generated.")

# ---Trajectory (early stages of progress)---------------------------------------------------------------------------
    def onGenerateReportsAndCSV(self):
        """Generate trajectory analysis reports and CSV files."""
        print("Generate Reports and CSV button clicked!")
        
        try:
            # Check if trajectory analysis is available
            if not TRAJECTORY_ANALYSIS_AVAILABLE:
                slicer.util.errorDisplay("Trajectory analysis module is not available. Please check the installation.")
                return
            
            # Get selected markup node
            markup_node = self.getSelectedMarkupNode()
            if not markup_node:
                slicer.util.errorDisplay("Please select a markup node containing electrode points.")
                return
            
            num_points = markup_node.GetNumberOfControlPoints()
            if num_points < 6:
                slicer.util.errorDisplay(f"Need at least 6 electrode points for trajectory analysis. Selected markup has {num_points} points.")
                return
            
            # Get trajectory IDs if specified
            trajectory_ids = self.parseTrajectoryIds()
            
            # Get output directory (patient folder)
            main_output_dir = self.getOutputDirectory()
            trajectory_dir = os.path.join(main_output_dir, "Trajectory_Analysis")
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # Get volume name from CT input or use markup name
            volume_name = markup_node.GetName()
            if hasattr(self.ui, 'inputSelectorCT') and self.ui.inputSelectorCT.currentNode():
                volume_name = self.ui.inputSelectorCT.currentNode().GetName()
            
            # Show progress
            with slicer.util.tryWithErrorDisplay("Failed to generate trajectory reports.", waitCursor=True):
                
                # Run trajectory analysis
                results = run_trajectory_analysis_on_markups(
                    markup_node=markup_node,
                    trajectory_ids=trajectory_ids,
                    output_dir=trajectory_dir,
                    volume_name=volume_name
                )
                
                if results['success']:
                    # Count generated files
                    generated_files = []
                    csv_file = os.path.join(trajectory_dir, f'trajectory_analysis_{volume_name}.csv')
                    viz_file = os.path.join(trajectory_dir, f'trajectory_visualization_{volume_name}.png')
                    html_file = os.path.join(trajectory_dir, f'trajectory_report_{volume_name}.html')
                    
                    if os.path.exists(csv_file):
                        generated_files.append("📊 Trajectory analysis CSV")
                    if os.path.exists(viz_file):
                        generated_files.append("📈 3D visualization plot")
                    if os.path.exists(html_file):
                        generated_files.append("🌐 Interactive HTML report")
                    
                    # Show success message
                    success_msg = f"""✅ Trajectory Reports Generated Successfully!

    📊 Analysis Results:
    - Markup node: {markup_node.GetName()}
    - Total electrode points: {results['total_points']}
    - Trajectories detected: {results['n_trajectories']}

    📁 Files saved to: {trajectory_dir}
    {chr(10).join([f'• {file}' for file in generated_files])}

    📈 CSV contains trajectory data with:
    - trajectory_id, n_contacts, length_mm, linearity_pca
    - center coordinates, spacing statistics  
    - curvature angles, algorithm scores"""
                    
                    if trajectory_ids:
                        success_msg += f"\n\n🔍 Analyzed specific trajectory IDs: {trajectory_ids}"
                    else:
                        success_msg += f"\n\n🔍 Analyzed all {results['n_trajectories']} detected trajectories"
                    
                    slicer.util.infoDisplay(success_msg)
                    
                else:
                    error_msg = f"❌ Trajectory analysis failed: {results.get('reason', 'Unknown error')}"
                    if 'available_ids' in results:
                        error_msg += f"\n\nAvailable trajectory IDs: {results['available_ids']}"
                    slicer.util.errorDisplay(error_msg)
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error during trajectory report generation: {str(e)}")
            logging.error(f"Trajectory report generation error: {str(e)}")

    def onCreateTrajectoryLines(self):
        """Create trajectory line markups in 3D Slicer."""
        print("Create Trajectory Lines button clicked!")
        
        try:
            # Check if trajectory analysis is available
            if not TRAJECTORY_ANALYSIS_AVAILABLE:
                slicer.util.errorDisplay("Trajectory analysis module is not available. Please check the installation.")
                return
            
            # Get selected markup node
            markup_node = self.getSelectedMarkupNode()
            if not markup_node:
                slicer.util.errorDisplay("Please select a markup node containing electrode points.")
                return
            
            num_points = markup_node.GetNumberOfControlPoints()
            if num_points < 6:
                slicer.util.errorDisplay(f"Need at least 6 electrode points for trajectory analysis. Selected markup has {num_points} points.")
                return
            
            # Get trajectory IDs - REQUIRED for line creation
            trajectory_ids = self.parseTrajectoryIds()
            if trajectory_ids is None or len(trajectory_ids) == 0:
                slicer.util.errorDisplay("Please specify trajectory IDs to create lines (e.g., 0,1,2)")
                return
            
            # Show progress
            with slicer.util.tryWithErrorDisplay("Failed to create trajectory lines.", waitCursor=True):
                
                # Run trajectory analysis (needed to get trajectory structure)
                results = run_trajectory_analysis_on_markups(
                    markup_node=markup_node,
                    trajectory_ids=trajectory_ids,
                    output_dir=None,  # Don't save files for line creation
                    volume_name="line_analysis"
                )
                
                if results['success']:
                    # Extract coordinates from markup node
                    coords_list = []
                    for i in range(num_points):
                        pos = [0, 0, 0]
                        markup_node.GetNthControlPointPosition(i, pos)
                        coords_list.append(pos)
                    coords_array = np.array(coords_list)
                    
                    # Create trajectory lines in Slicer
                    line_nodes = self.createTrajectoryLinesInSlicer(
                        results['analysis_results'], 
                        markup_node, 
                        coords_array,
                        trajectory_ids
                    )
                    
                    # Show success message
                    success_msg = f"""✅ Trajectory Lines Created Successfully!

    🎯 Generated Trajectory Lines:
    - Source markup: {markup_node.GetName()}
    - Number of trajectory lines: {len(line_nodes)}
    - Trajectory IDs: {trajectory_ids}
    - Color: Blue (for easy identification)

    📊 Analysis Summary:
    - Total electrode points used: {results['total_points']}
    - Trajectories detected: {results['n_trajectories']}

    💡 Lines are now visible in all 3D Slicer views."""
                    
                    slicer.util.infoDisplay(success_msg)
                    
                else:
                    error_msg = f"❌ Trajectory analysis failed: {results.get('reason', 'Unknown error')}"
                    if 'available_ids' in results:
                        error_msg += f"\n\nAvailable trajectory IDs: {results['available_ids']}"
                        error_msg += f"\nRequested IDs: {trajectory_ids}"
                    slicer.util.errorDisplay(error_msg)
            
        except Exception as e:
            slicer.util.errorDisplay(f"Error creating trajectory lines: {str(e)}")
            logging.error(f"Trajectory line creation error: {str(e)}")

    def parseTrajectoryIds(self):
        """Parse trajectory IDs from the input field."""
        trajectory_ids = None
        if hasattr(self.ui, 'trajectoryIdsLineEdit'):
            ids_text = self.ui.trajectoryIdsLineEdit.text.strip()
            if ids_text:
                try:
                    trajectory_ids = [int(x.strip()) for x in ids_text.split(',') if x.strip()]
                    print(f"Parsed trajectory IDs: {trajectory_ids}")
                except ValueError:
                    slicer.util.warningDisplay("Invalid trajectory IDs format. Please use format: 0,1,2")
                    return None
        return trajectory_ids

    def createTrajectoryLinesInSlicer(self, trajectory_results, markup_node, coords_array, requested_ids):
        """Create trajectory lines in 3D Slicer based on analysis results."""
        try:
            line_nodes = []
            
            if 'graph' not in trajectory_results:
                logging.warning("No graph data in trajectory results")
                return line_nodes
            
            # Get cluster assignments for each point
            clusters = np.array([node[1]['dbscan_cluster'] for node in trajectory_results['graph'].nodes(data=True)])
            
            # Create lines only for requested trajectory IDs
            for trajectory in trajectory_results['trajectories']:
                cluster_id = trajectory['cluster_id']
                
                # Skip if this trajectory ID was not requested
                if cluster_id not in requested_ids:
                    continue
                
                # Find points belonging to this trajectory
                cluster_mask = clusters == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) < 2:
                    logging.warning(f"Trajectory {cluster_id} has less than 2 points, skipping line creation")
                    continue
                
                # Get coordinates and sort along trajectory direction
                cluster_coords = coords_array[cluster_indices]
                
                # Sort points along the trajectory direction if available
                if 'direction' in trajectory and len(cluster_coords) > 2:
                    direction = np.array(trajectory['direction'])
                    center = np.mean(cluster_coords, axis=0)
                    projected = np.dot(cluster_coords - center, direction)
                    sorted_indices = np.argsort(projected)
                    cluster_coords = cluster_coords[sorted_indices]
                
                # Create curve markup node for complex trajectories
                if len(cluster_coords) > 2:
                    curve_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", 
                                                                f"Trajectory_Line_{cluster_id}")
                    
                    # Add all points to the curve
                    for i, coord in enumerate(cluster_coords):
                        curve_node.AddControlPoint(coord[0], coord[1], coord[2])
                    
                    # Set curve properties
                    curve_node.SetCurveTypeToLinear()  # Linear interpolation between points
                    
                    # Set blue color
                    display_node = curve_node.GetDisplayNode()
                    if display_node:
                        display_node.SetSelectedColor(0.0, 0.0, 1.0)  # Blue color (RGB)
                        display_node.SetLineThickness(0.4)
                        display_node.SetOpacity(1.0)
                        display_node.SetTextScale(0)  # Hide labels
                    
                    line_nodes.append(curve_node)
                    logging.info(f"Created blue trajectory curve for cluster {cluster_id} with {len(cluster_coords)} points")
                
                else:
                    # For 2-point trajectories, use line markup
                    line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", 
                                                                f"Trajectory_Line_{cluster_id}")
                    
                    # Add start and end points
                    line_node.SetNthControlPointPosition(0, cluster_coords[0][0], cluster_coords[0][1], cluster_coords[0][2])
                    line_node.SetNthControlPointPosition(1, cluster_coords[1][0], cluster_coords[1][1], cluster_coords[1][2])
                    
                    # Set blue color
                    display_node = line_node.GetDisplayNode()
                    if display_node:
                        display_node.SetColor(0.0, 0.0, 1.0)  # Blue color (RGB)
                        display_node.SetLineThickness(3.0)
                        display_node.SetOpacity(1.0)
                        display_node.SetTextScale(0)  # Hide labels
                    
                    line_nodes.append(line_node)
                    logging.info(f"Created blue trajectory line for cluster {cluster_id} with 2 points")
            
            return line_nodes
            
        except Exception as e:
            logging.error(f"Error creating trajectory lines: {e}")
            return []

#
# SEEG_ElectrodeLocalizationLogic
#

class SEEG_ElectrodeLocalizationLogic:
    """This class implements all the actual computation done by your module."""

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        # Initialize the brain mask extractor
        self.maskExtractor = BrainMaskExtractor()

    def process(self, inputVolume: vtkMRMLScalarVolumeNode, showResult: bool = True) -> vtkMRMLScalarVolumeNode:
        """
        Run the processing algorithm.
        Creates a mask using the BrainMaskExtractor.
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            The input volume to process
        showResult : bool, optional
            Whether to show the result in the Slicer viewer (default is True)
            
        Returns:
        --------
        vtkMRMLScalarVolumeNode
            The output volume node containing the mask
        """
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        # Use the brain mask extractor to create the mask
        return self.maskExtractor.extract_mask(inputVolume, threshold_value=20, show_result=showResult)

#
# SEEG_ElectrodeLocalizationTest
#

class SEEG_ElectrodeLocalizationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SEEG_masking1()

    def test_SEEG_masking1(self):
        """Test the module logic with sample data."""
        self.delayDisplay("Starting the test")

        # Get/create input data
        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample("SEEG_masking1")
        self.delayDisplay("Loaded test data set")

        # Test the module logic
        logic = SEEG_ElectrodeLocalizationLogic()
        
        # Process the input volume
        outputVolume = logic.process(inputVolume, True)
        
        # Check that output volume exists
        self.assertIsNotNone(outputVolume)
        
        # Basic validation that the output is a binary mask (0s and 1s only)
        outputArray = slicer.util.arrayFromVolume(outputVolume)
        unique_values = np.unique(outputArray)
        self.assertTrue(len(unique_values) <= 2, "Mask should only contain binary values")
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])), "Values should be 0 or 1")

        self.delayDisplay("Test passed")