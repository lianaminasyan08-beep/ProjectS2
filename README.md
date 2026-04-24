Lagrange Number Plate System

This project presents a C++ and Python–based approach for anonymizing vehicle license plate images using a mathematical method based on Lagrange interpolation. Instead of sharing the original license plate image, the system encodes the plate into a set of points and transformation parameters, which are used to generate a unique key. This key can later be used to reconstruct the structured representation of the plate without exposing the original image.

Technologies
C++ – core implementation for image processing and numerical computation
OpenCV – handling image operations (loading, contour extraction, transformations)
Python – supporting visualization and data handling
Streamlit – simple interface for displaying results
Lagrange Interpolation – used for reconstruction (decoding stage)
Features
Encoding license plate data into a mathematical representation
Generation of a unique key based on extracted points
Inclusion of transformation data (rotation and reflection) in the key
Reconstruction of plate structure using Lagrange interpolation
Visualization of curves representing character shapes
Privacy-preserving sharing (image + key separately)
How It Works
Encoding Stage
Upload an image of a license plate region
Extract contour points from characters
Apply rotation and reflection for normalization
Store extracted points and transformation parameters
Generate a key containing:
Lagrange points
Rotation and reflection angles (if applied)
Decoding Stage
Input the generated key
Apply inverse transformations
Use Lagrange interpolation to reconstruct the structure
Display the reconstructed plate as curves/plots