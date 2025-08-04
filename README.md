# Sentiment Analysis on Tweets using CNN-LSTM

This repository contains a Jupyter Notebook implementing a sentiment analysis model on tweets using a hybrid CNN-LSTM deep learning architecture. The model is trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) to classify tweets as positive or negative.

## Project Overview

The notebook performs sentiment analysis on a dataset of 1.6 million tweets, leveraging a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The pipeline includes data preprocessing, tokenization, model training, and evaluation, achieving an accuracy of approximately 82.2% on the test set.

### Key Features
- **Dataset**: Sentiment140 dataset with 1.6 million tweets.
- **Preprocessing**: Text cleaning, tokenization, and padding using NLTK and Keras.
- **Model**: Hybrid CNN-LSTM architecture with embedding, convolutional, pooling, LSTM, and dense layers.
- **Evaluation**: Accuracy, classification report, and confusion matrix visualization.
- **Technologies**: Python, TensorFlow, Keras, NLTK, Scikit-learn, Pandas, Matplotlib, Seaborn.

## Requirements

To run the notebook, ensure you have the following dependencies installed:

```bash
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
tensorflow
```

You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn tensorflow
```

Additionally, download the NLTK stopwords and punkt tokenizer:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Download the Dataset**:
   - The notebook uses the Sentiment140 dataset from Kaggle.
   - Download it manually from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) or use the Kaggle API:
     ```bash
     kaggle datasets download -d kazanova/sentiment140
     unzip sentiment140.zip
     ```
   - Ensure the dataset file (`training.1600000.processed.noemoticon.csv`) is placed in the project directory.

3. **Set Up Kaggle API** (if using Kaggle CLI):
   - Upload your `kaggle.json` API key to the project directory.
   - Set the Kaggle config directory:
     ```python
     import os
     os.environ['KAGGLE_CONFIG_DIR'] = "/path/to/your/project"
     ```
   - Secure the API key file:
     ```bash
     chmod 600 kaggle.json
     ```

## Usage

1. **Open the Notebook**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `CNN_LSTM_Based_Sentiment_Analysis_on_Tweets.ipynb`.

2. **Run the Notebook**:
   - Execute the cells sequentially to:
     - Load and preprocess the dataset.
     - Tokenize and pad the text data.
     - Build and train the CNN-LSTM model.
     - Evaluate the model and visualize results.

3. **Key Parameters**:
   - `max_features`: 5000 (number of words to consider in the vocabulary).
   - `max_words`: 50 (maximum length of sequences after padding).
   - `embed_dim`: 100 (embedding dimension).
   - `batch_size`: 64.
   - `epochs`: 3 (with early stopping).

4. **Output**:
   - The notebook outputs the model’s accuracy (82.2%), a classification report, and a confusion matrix plot.

## Model Architecture

The model combines CNN and LSTM layers:
- **Embedding Layer**: Converts words into dense vectors (100 dimensions).
- **Conv1D Layer**: 32 filters with kernel size 3 for feature extraction.
- **MaxPooling1D**: Reduces dimensionality while preserving important features.
- **LSTM Layer**: 50 units with dropout (0.2) for sequence modeling.
- **Dropout Layer**: 0.5 to prevent overfitting.
- **Dense Layer**: Softmax activation for binary classification (positive/negative).

## Results

- **Accuracy**: 82.2% on the test set.
- **Classification Report**: Provides precision, recall, and F1-score for both classes.
- **Confusion Matrix**: Visualizes true vs. predicted labels.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) for providing the data.
- TensorFlow and Keras for deep learning tools.
- NLTK and Scikit-learn for text preprocessing and evaluation.