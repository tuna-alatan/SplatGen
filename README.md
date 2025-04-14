# SplatGen

**SplatGen** is a command-line tool that takes a folder of input images, reconstructs a 3D scene using COLMAP, and outputs a 5-second panning video using Gaussian Splatting.

---

## 🚀 Features

- 🖼️ Takes a set of images from a scene (e.g., a mug)
- 📷 Automatically reconstructs camera poses and 3D structure using COLMAP
- 🎯 Extracts camera intrinsics & extrinsics for all registered images
- 🧠 (Soon) Trains Gaussian splats using the reconstructed scene
- 🎥 Generates a smooth 5-second orbit animation of the scene
- 📦 CLI-first, easy to extend and integrate

