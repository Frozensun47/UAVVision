# UAVVision : Efficient Object Detection for Resource-Constrained Drones using YOLO Conversion and NGWD Enhancement
# UAV Object Detection Enhancement Project


![](https://github.com/Frozensun47/UAVVision/blob/main/Extras/UAVVision_LOGO_GIF.gif)


## Introduction

The **UAV Object Detection Enhancement Project** is a pioneering initiative aimed at improving object detection performance for Unmanned Aerial Vehicles (UAVs) or drones, equipped with limited computational resources, specifically Raspberry Pi boards. The project utilizes cutting-edge technologies including the NCNN (Neural Network Computer Vision) library, YOLO (You Only Look Once) model architecture, and a novel approach involving image patching to enhance accuracy in detecting small-scale objects.

## Problem Statement

Object detection tasks are inherently complex and computationally intensive, particularly when constrained by the processing power available on drones. The primary challenge lies in achieving accurate real-time object detection while ensuring optimal resource utilization.

## Technologies Used

- **NCNN Library**: The NCNN library serves as the backbone of the project, offering a lightweight framework optimized for mobile platforms and embedded systems. Its efficiency in utilizing hardware acceleration, such as GPUs and neural network accelerators, enables real-time inference on resource-constrained devices.

- **YOLO Model Architecture**: The project adopts the YOLO model architecture due to its ability to perform object detection in a single forward pass, making it highly efficient for real-time applications. The model is customized and optimized for the target platform, striking a balance between accuracy and computational efficiency.

- **Image Patching**: To address the challenge of detecting small objects, the innovative strategy of image patching is employed. High-resolution frames captured by the drone's Raspberry Pi camera (1920x1080) are divided into smaller patches (e.g., 240x135). This approach increases the likelihood of capturing small objects while maintaining a manageable computational load.

- **Normalized Gaussian Wasserstein Distance Loss**: The project leverages the Normalized Gaussian Wasserstein Distance Loss function, which enhances the model's ability to accurately localize and classify objects, particularly those at smaller scales. This specialized loss function contributes to better object detection performance.

## Workflow and Methodology

1. **Data Acquisition**: The drone captures high-definition (1920x1080) video using the Raspberry Pi camera. The video frames are extracted to form the input data for the object detection process.

2. **Image Patching**: Each high-resolution frame is divided into multiple patches, e.g., 16 patches of 240x135 dimensions. This division increases the likelihood of detecting small objects within the scene.

3. **Model Inference**: The YOLO model, tailored for the target hardware, processes each image patch independently to predict object bounding box coordinates (center, height, width) and class labels.

4. **Post-processing**: The predicted object information from the patches is integrated to form a coherent output representing the objects detected within the entire frame.

5. **Performance Enhancement**: The utilization of the Normalized Gaussian Wasserstein Distance Loss enhances the model's capacity to accurately detect and classify objects, especially those of smaller scales.

6. **Real-time Visualization**: The processed video frames, now with bounding box annotations for detected objects, are presented in real-time, providing valuable insights to drone operators.

## Achievements

The project's novel approach and judicious utilization of technologies yield impressive results:

- **Mean Average Precision (mAP)**: The project achieves a mAP of 24.1, showcasing its accuracy in detecting and localizing objects within the frames.

- **Frame Rate Performance**: The system operates at a commendable 18.8 Frames Per Second (FPS) on a Raspberry Pi 4 clocked at 1950 MHz. This high frame rate is crucial for real-time applications, allowing quick responses to dynamic environments.

## Conclusion

The **UAV Object Detection Enhancement Project** demonstrates a pioneering approach to address the challenges of real-time object detection on resource-constrained UAVs. By harnessing the power of the NCNN library, customizing the YOLO model architecture, implementing image patching, and utilizing specialized loss functions, the project achieves impressive accuracy and frame rate performance. This advancement opens up new horizons for UAV applications, such as surveillance, environmental monitoring, and disaster response, where real-time and accurate object detection is of paramount importance.

*Note: The project was completed independently and reflects a strong blend of creativity, technical prowess, and problem-solving skills.*

---
"Pushing the boundaries of object detection in resource-constrained environments."
