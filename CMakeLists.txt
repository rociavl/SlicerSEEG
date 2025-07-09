cmake_minimum_required(VERSION 3.16.3...3.19.7)

project(SlicerSEEG)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/rociavl/SlicerSEEG")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Rocio Avalos (UPC), Alessandro Principe (Hospital del Mar), Justo Montoya-Galvez (UPF)")
set(EXTENSION_DESCRIPTION "Automated SEEG electrode localization system for epilepsy surgery planning. Transforms manual 4+ hour electrode identification into 30-minute automated workflow with confidence-based validation.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/SEEG_masking/Resources/Icons/SEEG_masking.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/Screenshots/workflow_overview.png")
set(EXTENSION_DEPENDS "NA") # Dependent extensions

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SEEG_masking)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})