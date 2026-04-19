# 2D Matrix Structural Analysis Engine

## Overview
This project is a fully object-oriented 2D Matrix Structural Analysis engine written entirely from scratch in Python. It is designed to analyze complex combinations of frame and truss elements subjected to advanced mechanical boundary conditions, without relying on external mathematical libraries like NumPy or SciPy.

## Technical Features
* Custom Matrix Engine: Uses a custom-built matrix_library.py featuring a memory-efficient Sparse Matrix solver utilizing the Dictionary of Keys (DOK) scheme.
* Mixed Element Systems: Seamlessly solves mixed systems of FrameElement and TrussElement objects, automatically handling rotational degree-of-freedom locking to prevent matrix singularities.
* Statical Condensation: Supports member-end releases (internal hinges) by mathematically condensing the local stiffness matrices and redistributing fixed-end forces.
* Support Settlements: Capable of calculating internal forces induced by prescribed support displacements (dx, dy, rz).
* Thermal Loading: Handles both uniform temperature changes (axial expansion) and thermal gradients across member depths (thermal bending).
* Axis Transformation (Rigid Offsets): Features independent start and end physical offsets (offset_start_y, offset_end_y). The engine mathematically shifts neutral axis stiffness and fixed-end forces to physical connection points using a Rigid Transformation Matrix.

## File Structure
* main.py: The entry point. Parses the input.txt file, builds the object-oriented system, executes the analysis, and writes the results.
* frame_solver.py: The core physics engine. Contains the classes for Elements, Nodes, Materials, Sections, Loads, and the StructuralModel assembly/solving logic.
* matrix_library.py: Custom mathematical library for sparse and dense matrix operations.
* test_frame_solver.py: A comprehensive automated testing suite featuring 22 unit and integration tests to verify mechanical physics and catch user-input validation errors.
* input.txt: The user-defined structural model and loading conditions.
* output_report.txt: The generated analysis report.

## Usage

1. Define the Structure
Edit the input.txt file to define your materials, nodes, members, supports, and loads. The parser supports scientific notation (e.g., 1.2e-5) for physical properties and accepts Modulus of Elasticity in GPa.

2. Run the Analysis
Execute the main script from your terminal:
python main.py

3. View the Results
The program will generate an output_report.txt file containing equation numbering, nodal displacements, local member end forces, and global support reactions.

## Testing
To verify the engine's integrity using the built-in test suite, run the following command in your terminal:
python -m unittest test_frame_solver.py -v

## Assignment 4 Implementations
* Structure A: Verification of a mixed concrete-frame and steel-truss structure subjected to a 2 mm downward support settlement.
* Structure B: Verification of a thermally loaded structure utilizing the Rigid Offset transformation to physically model trusses connecting 400 mm below the neutral axis of an 80 cm deep concrete beam.
