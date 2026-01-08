# Sign Language Recognition Project

## Overview
This project aims to develop a deep learning model for recognizing hand gestures representing numbers in sign language. The system uses the AlexNet architecture and custom variations to classify images into 10 classes (digits 0-9).

## Project Structure
```
project_sign_language/
├── database/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   ├── 7/
│   ├── 8/
│   ├── 9/
├── model/
│   ├── alexnet_sign_language_model.h5
│   ├── alexnet_sign_language_model1.h5
│   ├── ori_alexnet_sign_language_model.h5
├── notebook/
│   ├── main.ipynb
├── requirements.txt
└── README.md
```

## Features
- **Database**: Contains images of hand gestures for digits 0-9.
- **Models**: Pre-trained and custom-trained AlexNet models.
- **Notebook**: Jupyter Notebook for training, evaluation, and visualization.

## Requirements
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

### Dependencies
- opencv-python
- numpy
- tensorflow
- matplotlib
- Pillow
- scikit-learn
- seaborn
- pandas
- h5py
- jupyter
- tensorflow-hub

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project_sign_language
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebook/main.ipynb
   ```

4. Follow the steps in the notebook to train and evaluate the model.

## Models
### AlexNet Original
- Architecture: 5 convolutional layers, 3 fully connected layers.
- Optimized for sparse categorical cross-entropy loss.

### Custom AlexNet Variations
- Modified convolutional and dense layers for improved performance.

## Contributions
Team members can contribute by:
- Adding new datasets.
- Improving model architectures.
- Enhancing visualization and evaluation metrics.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

