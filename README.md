# UAVVision : Efficient Object Detection for Resource-Constrained Drones using YOLO Conversion and NGWD Enhancement

## Object Detection on UAVs using YOLO Model Conversion and NGWD Loss

In this project, I successfully implemented an innovative solution for real-time object detection on Unmanned Aerial Vehicles (UAVs) using a YOLO (You Only Look Once) model converted to the ncnn framework. The primary goal was to enable efficient object detection on resource-constrained platforms such as Raspberry Pi, which are commonly deployed on drones and UAVs.

### Project Highlights:

- **Model Conversion and Optimization**: I utilized the ncnn framework to convert the YOLO model, optimizing it for deployment on Raspberry Pi. This conversion process involved adapting the model to the target hardware, ensuring optimal performance without compromising accuracy.

- **Object Detection**: I implemented a robust object detection pipeline using the converted model. The model was capable of accurately detecting and localizing objects of interest in real-time video streams captured by drones.

- **Handling Small Objects**: To address the challenge of detecting small objects in high-resolution images, I adopted an innovative approach. I divided large 1920x1080 images into 16 smaller images of 240x135 resolution. This technique significantly improved the model's ability to capture small objects within the scene.

- **Normalized Gaussian Wasserstein Distance (NGWD) Loss**: To further enhance the detection of small objects, I incorporated the NGWD loss function. This specialized loss function helped the model focus on the details of small objects, leading to improved accuracy in their detection.

- **Performance Optimization**: I fine-tuned the model and its parameters to ensure optimal performance on the Raspberry Pi. This involved a balance between accuracy and real-time processing, considering the hardware limitations of the UAVs.

- **Real-world Impact**: The project has practical applications in surveillance, search and rescue operations, and environmental monitoring, where drones equipped with accurate object detection capabilities can provide critical insights and decision-making support.

### Technologies and Tools Used:

- YOLO Model
- ncnn Framework
- Raspberry Pi
- Image Processing
- Machine Learning
- Python
- Normalized Gaussian Wasserstein Distance Loss
- Object Localization
- Performance Optimization

This project underscores my ability to adapt and optimize machine learning models for deployment in resource-constrained environments. The successful implementation of object detection on UAVs reflects my strong skills in computer vision, model conversion, and creative problem-solving.
