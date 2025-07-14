cmake_minimum_required(VERSION 3.16.3...3.19.7)

project(SlicerSEEG)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.example.com/SEEG")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Rocio Avalos (UPC)")
set(EXTENSION_DESCRIPTION "Semi-Automated SEEG electrode localization for epilepsy surgery planning")
set(EXTENSION_ICONURL "SEEG_masking/Resources/Icons/SEEG_masking.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/rociavl/SlicerSEEG/main/Screenshots/workflow_overview.png")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SEEG_ElectrodeLocalization)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})