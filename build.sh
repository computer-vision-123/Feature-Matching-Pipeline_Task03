#!/bin/bash

# Exit immediately if any command fails
set -e

echo "--- 1. Detecting active Python environment ---"
PYTHON_EXE=$(which python)
echo "Using Python: $PYTHON_EXE"

echo "--- 2. Cleaning old build artifacts ---"
rm -rf build

echo "--- 3. Creating fresh build directory ---"
mkdir build
cd build

echo "--- 4. Configuring CMake ---"
# Note: Modern CMake uses Python_EXECUTABLE instead of PYTHON_EXECUTABLE
# Get pybind11's cmake directory from the active Python environment
PYBIND11_DIR=$($PYTHON_EXE -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "Using pybind11 cmake dir: $PYBIND11_DIR"
cmake -DPython_EXECUTABLE="$PYTHON_EXE" -Dpybind11_DIR="$PYBIND11_DIR" ..

echo "--- 5. Compiling backend ---"
# Use all available CPU cores for a faster build
make -j$(nproc)

# Go back to the root directory
cd ..

echo "---------------------------------------------------"
echo "Build complete! The module is ready in the 'build' folder."
echo "You can now run your app: python Frontend/Main_window.py"
echo "---------------------------------------------------"