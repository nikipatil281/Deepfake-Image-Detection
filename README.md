# DeepTruth - Unmasking the Unreal  

## 📝 Description 
DeepTruth is a deep learning-based framework designed to **detect and differentiate real human faces from AI-generated fakes**.  
With the rise of **DeepFakes** and synthetic media, the ability to distinguish real from fake has become crucial in fighting misinformation, identity fraud, and digital manipulation.  

DeepTruth leverages **state-of-the-art deep learning models** trained on diverse datasets containing both **real and synthetic faces** to create a highly accurate and robust detection system.  

## 📚 Table of Contents  
- [DeepTruth - Unmasking the Unreal](#deeptruth---unmasking-the-unreal)
  - [📝 Description](#-description)
  - [📚 Table of Contents](#-table-of-contents)
  - [🗂️ Data Sources](#️-data-sources)
    - [**Real Face Datasets:**](#real-face-datasets)
    - [**Synthetic (AI-Generated) Face Datasets:**](#synthetic-ai-generated-face-datasets)
  - [Features](#features)
  - [📦 Installation](#-installation)
  - [⚙️ Usage](#️-usage)
    - [🔧 Training the Model](#-training-the-model)
    - [🧪 Testing with Pretrained Model](#-testing-with-pretrained-model)
  - [🧰 Dependencies](#-dependencies)
  - [📬 Contact](#-contact)

##  🗂️ Data Sources  
DeepTruth is trained and evaluated on a combination of **real and synthetic face datasets** to ensure robust performance.  

### **Real Face Datasets:**  
- **CelebA (CelebFaces Attributes Dataset)**  
  - Contains over **200,000 celebrity images** with **40 attribute annotations**.  

- **FFHQ (Flickr-Faces-HQ)**  
  - A high-quality dataset containing **70,000 PNG images** at **1024×1024 resolution**.  
  

- **Human Faces (Kaggle)**  
  - A collection of **real human face images**.  

### **Synthetic (AI-Generated) Face Datasets:**  
- **DeepFake Face Detection Dataset (Kaggle)**  
  - A dataset containing **DeepFake-generated images** for fake face recognition.  
  
- **StyleGAN3 Generated Images**  
  - Custom **synthetic face images generated using StyleGAN3**.  

- **Generated Faces from Online Sources**  
  - Additional AI-generated face images sourced from:  ThisPersonDoesNotExist  

## Features

- **Excellent Generalization** across unseen DeepFake types and datasets  
  - No fully pre-trained deep learning model was used  
  - Instead, a partially pre-trained CNN was customized: **some layers were frozen**, while others were fine-tuned to suit the task  
- **Deep Learning Model** (CNN based)  
- Trained on Large-Scale Real & Synthetic Face Datasets 
- Image input size to the models = **512 x 512** 
- Supports **Image DeepFake Detection**  
- Achieves **High Accuracy** in Detecting AI-Generated Faces  
- Compatible with Multiple DeepFake Generators *(StyleGAN, FaceSwap, etc.)*  
- **Open-Source** & Extendable for Future Research  


## 📦 Installation

Clone the repository or download the raw files:
```bash
git clone https://github.com/nikipatil281/Deepfake-Image-Detection.git
cd Deepfake-Image-Detection
```

(Optional but recommended) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Usage

### 🔧 Training the Model

To train the model from scratch:
- Open and run the Jupyter notebook: `notebooks/train_model.ipynb`
- Make sure the dataset paths are correctly set inside the notebook.
- Make sure the `DESIRED PATH TO MODEL` is replaced with the actual path the trained model should be saved
- You’ll see training progress, accuracy score and losses.
- The trained model will be saved automatically (as a `.pth` file).

### 🧪 Testing with Pretrained Model

To test our pretrained model:
- Load the notebook: `notebooks/Predict_real_or_fake.ipynb`
- Upload an image file (real or fake) by providing the path of the image in the notebook - Where ever `PATH TO INPUT IMAGE` is mentioned, add the image path
- Make sure the model path (`model/1.8L_deepfake_detector.pth`) is correctly set inside the notebook.
- The notebook will load the saved model and display whether the image is real or fake.
---

## 🧰 Dependencies  
All required packages are listed in `requirements.txt`. Core packages include:
- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- opencv-python
- IPython

## 📬 Contact  
For questions, collaborations, or issues, reach out to:

|                 | **Nikita Patil**                                        | **Ajitesh K**                                               |
|-----------------|---------------------------------------------------------|-------------------------------------------------------------|
| 📧 Email        | [nikipatil281@gmail.com](mailto:nikipatil281@gmail.com) | [ajiteshkanumuru394@gmail.com](mailto:ajiteshkanumuru394@gmail.com) |
| 🔗 GitHub       | [nikipatil281](https://github.com/nikipatil281)         | [Ajitesh](https://github.com/ajiteshkanumuru)                 |
| 🔗 LinkedIn     | [LinkedIn](https://www.linkedin.com/in/nikita-patil-niki) | [LinkedIn](https://www.linkedin.com/in/ajitesh-k394/) |

---
