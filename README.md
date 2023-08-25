# Face Mask Detection Using Computer Vision
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![Face Mask Detection](images/demo.gif)

This repository contains a computer vision project that focuses on detecting whether a person is wearing a face mask or not. The project utilizes state-of-the-art deep learning techniques to analyze images or video streams from various sources, such as webcams or recorded videos, and provides real-time feedback on whether individuals in the frames are wearing masks or not.



## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Future Enhancements](#future-enhancements)
- [Contributions](#contributions)
- [License](#license)

## Introduction

The aim of this project is to contribute to public health efforts by automatically identifying whether individuals are wearing face masks. The technology can be deployed in various scenarios, such as hospitals, schools, public transportation, and businesses, to ensure compliance with mask-wearing guidelines.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the face mask detection system:

1. Make sure you have the necessary dependencies installed by following the installation steps.

2. Run the detection script:

   ```bash
   python detect_mask.py
   ```

3. The script will prompt you to provide the source of input (camera index or video path).

4. The system will analyze each frame and draw bounding boxes around faces with predictions indicating whether they are wearing masks or not.

## Model

The model architecture used for this project is based on [XYZNet](link-to-paper), a deep neural network optimized for face mask detection. The model is pretrained on a large dataset and fine-tuned on our custom dataset.

## Dataset

The dataset used for training and evaluation consists of images of individuals with and without face masks. It was collected from various sources and annotated manually. The dataset is divided into training, validation, and testing subsets.

## Training

To train the model:

1. Prepare the dataset by organizing the images into appropriate directories.

2. Run the training script:

   ```bash
   python train.py --dataset /path/to/dataset --epochs 20
   ```

3. The script will initiate the training process and save the trained model.

## Evaluation

To evaluate the model:

1. Run the evaluation script:

   ```bash
   python evaluate.py --model /path/to/model --dataset /path/to/dataset
   ```

2. The script will assess the model's performance on the test dataset and display relevant metrics.

## Demo

Check out our [demo video](demo/demo.mp4) showcasing the real-time face mask detection system in action.

## Future Enhancements

We are actively working on improving the project:

- Multi-person detection.
- Mask type classification (surgical, N95, cloth, etc.).
- Integration with access control systems.

## Contributions

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize the content in this `README.md` to match your project's specifics. Include any additional sections or information that you find relevant. Good luck with your face mask detection project!

# Output


https://github.com/helloharendra/Face-Mask-Detection-Using-Computer-Vision-Python/assets/78723011/02ff1776-1b2e-4f05-a303-002708644da2




# User Interface

![Screenshot 2022-10-12 at 2 09 27 AM](https://user-images.githubusercontent.com/78723011/195432070-1f361799-6455-4127-b586-938a74b7e53a.png)
![Screenshot 2022-10-12 at 2 07 07 AM](https://user-images.githubusercontent.com/78723011/195432169-d577b36c-38e3-445a-9213-d9865b6a11df.png)
![Screenshot 2022-10-12 at 2 07 14 AM](https://user-images.githubusercontent.com/78723011/195432206-0b608690-603e-4818-9bed-df288b5b44d7.png)
![Screenshot 2022-10-12 at 2 07 25 AM](https://user-images.githubusercontent.com/78723011/195432209-ea39daf6-f69b-4379-9860-bfec29376136.png)
![Screenshot 2022-10-12 at 2 07 32 AM](https://user-images.githubusercontent.com/78723011/195432216-d4d012bf-2400-4d4b-8273-8739c6198488.png)
![Screenshot 2022-10-12 at 2 07 46 AM](https://user-images.githubusercontent.com/78723011/195432223-3964d653-c07a-4415-b4fd-c33a6ad6c481.png)
![Screenshot 2022-10-12 at 2 07 53 AM](https://user-images.githubusercontent.com/78723011/195432233-b20188ff-2793-46dc-8dcd-6de1033a4b8b.png)
![Screenshot 2022-10-12 at 2 08 03 AM](https://user-images.githubusercontent.com/78723011/195432786-59446fbe-5720-4d01-b7dc-ccb4e5ef33f2.png)
![Screenshot 2022-10-12 at 2 08 17 AM](https://user-images.githubusercontent.com/78723011/195432814-1c731259-5878-42d2-a24a-13cfbcf2dc1b.png)
![Screenshot 2022-10-12 at 2 08 21 AM](https://user-images.githubusercontent.com/78723011/195432818-249d2707-8bed-44cc-b321-3301134c5d0d.png)

