# Unsupervised Learning: Credit Card Fraud Detection

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/ahmedjawed/creditscardfrauddataset)

Demonstration of unsupervised machine learning techniques for both image segmentation and fraud detection.

## Project Overview

This project explores unsupervised learning algorithms through two practical applications:

1. **Image Segmentation with Clustering**
   - K-Means clustering implementation from scratch
   - K-Medoids clustering implementation
   - Color-based image segmentation using YCbCr color space

2. **Credit Card Fraud Detection using Anomaly Detection**
   - Isolation Forest algorithm
   - Local Outlier Factor (LOF) algorithm
   - PCA and t-SNE for dimensionality reduction and visualization
   - Comparison of different detection methods

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/ahmedjawed/creditscardfrauddataset) which contains transactions made by European cardholders. The dataset is highly imbalanced, with only 0.17% of transactions being fraudulent.

**Note:** Due to the large size (~150MB), the dataset file (`creditcard.csv`) is not included in this repository. Please download it from Kaggle.

## Key Features

- Custom implementations of K-Means and K-Medoids algorithms
- Visualization of clustering results
- Comparative analysis of anomaly detection methods
- Performance evaluation using precision, recall, and F1-score
- Dimensionality reduction for visualization

## Technologies Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV (cv2)
- PIL (Pillow)

## Getting Started

1. Clone this repository
2. Download the `creditcard.csv` dataset from Kaggle and place it in the project root
3. Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn opencv-python pillow`
4. Open and run the Jupyter notebook

## Results

The notebook includes:
- Visual comparisons of clustering algorithms on image segmentation
- ROC curves and performance metrics for fraud detection
- t-SNE visualizations of detected anomalies

## License

This project is for educational and portfolio purposes.
