# OpenCV – Computer Vision Experiments with Python

A collection of hands-on Computer Vision experiments and practical implementations using **Python, OpenCV, NumPy, and MediaPipe**. This repository covers image processing, face detection, face landmarks, human pose estimation, perspective transformation, and video processing.

The goal of this project is to understand fundamental and advanced Computer Vision concepts through practical Python implementations.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Experiments](#-experiments)
- [Technologies and Libraries](#-technologies-and-libraries)
- [Prerequisites](#-prerequisites)
- [Installation and Setup](#-installation-and-setup)
- [Running the Experiments](#-running-the-experiments)
- [Webcam and Video Requirements](#-webcam-and-video-requirements)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📌 Project Overview

Computer Vision is a field of Artificial Intelligence that enables computers to interpret and understand visual information from images and videos.

This project contains multiple practical experiments developed using Python and Computer Vision libraries.

It demonstrates how to:

- Capture images and videos using a webcam.
- Detect human faces using Haar Cascade classifiers.
- Perform face detection using MediaPipe.
- Identify facial landmarks using Face Mesh.
- Perform human pose estimation.
- Process and manipulate digital images.
- Apply perspective transformations.
- Calculate and display video frame rates (FPS).

---

## ✨ Features

- Real-time face detection.
- Haar Cascade-based face recognition and detection.
- MediaPipe-based face detection.
- Face Mesh and facial landmark visualization.
- Human pose detection and landmark tracking.
- Image processing operations.
- Perspective transformation.
- Webcam integration.
- Video processing and FPS calculation.
- Real-time visual output using OpenCV.

---

## 🧪 Experiments

The repository contains the following experiment categories:

### 1. Face Detection using Haar Cascade

- Detects human faces from webcam input.
- Uses a pre-trained Haar Cascade classifier.
- Draws rectangular bounding boxes around detected faces.

**Technologies:** OpenCV, Haar Cascade XML.

### 2. Face Detection using MediaPipe

- Detects faces in real-time.
- Uses Google's MediaPipe framework.
- Displays face detection results through OpenCV.

**Technologies:** MediaPipe, OpenCV.

### 3. Face Mesh Detection

- Detects facial landmarks.
- Identifies detailed facial structures.
- Visualizes facial landmark connections.

**Technologies:** MediaPipe Face Mesh, OpenCV.

### 4. Human Pose Detection

- Detects human body landmarks.
- Tracks body positions and movements.
- Visualizes pose landmarks and connections.

**Technologies:** MediaPipe Pose, OpenCV.

### 5. Image Processing

- Performs image manipulation and processing.
- Demonstrates fundamental image processing techniques.
- Uses OpenCV for image reading, display, and processing.

**Technologies:** OpenCV, NumPy.

### 6. Perspective Transformation

- Demonstrates perspective transformation.
- Converts images between different viewing perspectives.
- Uses transformation matrices and image warping.

**Technologies:** OpenCV, NumPy.

### 7. Video Processing and FPS Calculation

- Reads and processes video frames.
- Calculates video frame rate (FPS).
- Displays FPS information during video processing.

**Technologies:** OpenCV, Python.

### 8. Additional Camera and Image Experiments

- Webcam capture.
- Image loading and display.
- Basic computer vision operations.
- Practical Python implementations.

**Technologies:** OpenCV, NumPy.

---

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing |
| MediaPipe | Face and pose detection |
| NumPy | Numerical and array operations |
| Matplotlib | Image visualization and plotting |

### Install the required libraries

Open your terminal and execute:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python
python -m pip install mediapipe
python -m pip install numpy
python -m pip install matplotlib
```

### Install all libraries together

```bash
pip install opencv-python mediapipe numpy matplotlib
```

**Note:** These are the core dependencies for the visible experiments. Additional dependencies may be required by individual scripts.

---

## 💻 Prerequisites

Before running this project, make sure you have:

- Windows 10 or Windows 11.
- Python 3.12 recommended.
- Visual Studio Code or another Python IDE.
- Git installed.
- A working webcam for real-time detection experiments.
- Internet connection for installing dependencies.

Check your Python installation:

```bash
python --version
```

Check pip:

```bash
pip --version
```

---

## ⚙️ Installation and Setup

Follow these steps to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/rishabhcseaiml/openCv_project.git
```

### Step 2: Navigate to the Project

```bash
cd openCv_project
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 6: Install Dependencies

```bash
pip install opencv-python mediapipe numpy matplotlib
```

### Step 7: Verify Installation

Run:

```bash
python -c "import cv2, mediapipe, numpy, matplotlib; print('All libraries installed successfully!')"
```

If the command executes without errors, your environment is ready.

---

## ▶️ Running the Experiments

Open the repository in Visual Studio Code:

```bash
code .
```

Activate your virtual environment before executing the scripts.

### Run individual experiments

**Haar Cascade Face Detection**

```bash
python Experiment9_FaceDetection/haar_face_detection.py.py
```

**MediaPipe Face Detection**

```bash
python Experiment9_FaceDetection/mediapipe_face_detection.py
```

**Face Mesh Detection**

```bash
python FaceMesh_Experiments/face_mesh.py
```

**Basic Camera Experiment**

```bash
python camera.py
```

**Image Processing Experiment**

```bash
python exp4.py
```

**FPS Calculation**

```bash
python fps.py
```

**Additional FPS Experiments**

```bash
python fps2.py
python fps3.py
```

**Important:** The commands above use filenames visible in the project screenshots. Check the actual spelling and folder names in your cloned repository before running them. For other experiments, navigate to the respective folder and execute its Python script.

---

## 📷 Webcam and Video Requirements

Some experiments require a webcam to capture real-time images.

### Webcam Requirements

- Working laptop or external USB webcam.
- Camera permissions enabled in Windows.
- Sufficient lighting for face and pose detection.
- Close other applications currently using the camera.

### Webcam Test

Create a Python file named `webcam_test.py`:

```python
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run:

```bash
python webcam_test.py
```

Press **Q** to close the webcam window.

### Video Requirements

For video processing experiments:

- Use a supported video format such as MP4.
- Ensure the video file path is correct.
- Make sure the video file is accessible.
- Verify that the required video codecs are installed.

---

## 📁 Project Structure

The following is a representative structure based on the visible repository folders. Some names may need adjustment to match your local files.

```text
openCv_project/
│
├── Experiment8_Camera/
│
├── Experiment9_FaceDetection/
│   ├── haar_face_detection.py.py
│   ├── mediapipe_face_detection.py
│   └── haarcascade_frontalface_default.xml
│
├── Experiment10_YOLO/
│
├── FaceMesh_Experiments/
│   └── face_mesh.py
│
├── Human-Pose-Detection/
│
├── ImageProcessing/
│
├── Perspective_Transformation/
│
├── npl/
│
├── camera.py
├── exp4.py
├── EXP5.py
├── fps.py
├── fps2.py
├── fps3.py
│
├── .gitignore
└── README.md
```

---

## 🐛 Troubleshooting

### 1. ModuleNotFoundError

**Problem:**

```text
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**

```bash
python -m pip install opencv-python
```

For MediaPipe:

```bash
python -m pip install mediapipe
```

Make sure VS Code is using the same Python interpreter where you installed the libraries.

### 2. MediaPipe Installation Error

Try using Python 3.12 and create a fresh virtual environment:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install mediapipe opencv-python numpy
```

### 3. Webcam Not Opening

Check whether another application is using your webcam.

Try changing:

```python
cap = cv2.VideoCapture(0)
```

To:

```python
cap = cv2.VideoCapture(1)
```

Also verify Windows camera permissions.

### 4. Haar Cascade File Not Found

**Problem:**

```text
CascadeClassifier returned empty classifier
```

**Solution:**

Check that the XML file exists and use the correct path:

```python
face_cascade = cv2.CascadeClassifier(
    "Experiment9_FaceDetection/haarcascade_frontalface_default.xml"
)
```

### 5. Video File Not Opening

Check the video
