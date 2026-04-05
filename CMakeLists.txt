cmake_minimum_required(VERSION 3.15)
project(ShapeAndContourAnalyzer VERSION 1.0 LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==============================================================================
# 1. DEPENDENCIES
# ==============================================================================

# Find the exact Python version you are currently running
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find Pybind11 (Make sure your virtual environment is activated before running cmake!)
find_package(pybind11 REQUIRED)

# Find OpenCV (Required for your actual image processing later)
find_package(OpenCV REQUIRED)

# ==============================================================================
# 2. CREATE THE PYTHON MODULE
# ==============================================================================

# Create the module and name it "cv_backend" (this MUST match your Python import)
# Add all your C++ source files here
pybind11_add_module(cv_backend 
    Backend/Main_bindings.cpp
    Backend/Extraction.cpp
    Backend/Description.cpp
    Backend/Matching.cpp
)

# Tell the compiler where to look for header files (like your image_utils.hpp)
target_include_directories(cv_backend PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/Backend
    ${CMAKE_CURRENT_SOURCE_DIR}/Helpers
    ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV to your backend
target_link_libraries(cv_backend PRIVATE ${OpenCV_LIBS})

# ==============================================================================
# 3. CROSS-PLATFORM OUTPUT FIX
# ==============================================================================

# Force the compiled module (.so on Linux, .pyd on Windows) to output 
# directly into the 'build' folder instead of nested 'Release'/'Debug' folders.
set_target_properties(cv_backend PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Extra enforcement for multi-config generators like Windows Visual Studio
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set_target_properties(cv_backend PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_CURRENT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_CURRENT_BINARY_DIR}"
    )
endforeach()