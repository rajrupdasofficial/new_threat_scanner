# Deep Learning Based Vulnerability Scanner

A sophisticated deep learning system for automated vulnerability detection and classification using TensorFlow and Apache Spark. This project classifies CVE (Common Vulnerabilities and Exposures) descriptions into CWE (Common Weakness Enumeration) categories with 86.9% accuracy.

## 🎯 Project Overview

The Deep Learning Based Vulnerability Scanner leverages advanced neural network architectures to automatically analyze and categorize security vulnerabilities. Built with pure deep learning libraries (TensorFlow + Apache Spark), it processes large-scale vulnerability datasets without relying on traditional machine learning frameworks like sklearn or pandas.

## ✨ Key Features

- **High Accuracy**: Achieves 86.9% test accuracy on vulnerability classification
- **Scalable Architecture**: Handles datasets with 50M+ records using Apache Spark
- **GPU Accelerated**: Optimized for NVIDIA GPUs (tested on RTX 3050)
- **Deep Neural Network**: Multi-layer bidirectional LSTM with dense layers
- **Memory Optimized**: Intelligent data sampling and chunked processing
- **Pure Deep Learning**: Built exclusively with TensorFlow and Spark (no sklearn/pandas)

## 🏗️ Architecture

### Neural Network Structure
```
Input Layer (Text Sequences)
    ↓
Embedding Layer (256 dimensions, 10K vocabulary)
    ↓
Bidirectional LSTM (256 units) + Dropout (0.3)
    ↓
Bidirectional LSTM (128 units) + Dropout (0.3)
    ↓
Bidirectional LSTM (64 units) + Dropout (0.3)
    ↓
Dense Layer (512 units) + Dropout (0.4)
    ↓
Dense Layer (256 units) + Dropout (0.4)
    ↓
Dense Layer (128 units) + Dropout (0.4)
    ↓
Dense Layer (64 units)
    ↓
Output Layer (Softmax - CWE Categories)
```

### Data Processing Pipeline
1. **Data Loading**: Apache Spark CSV processing with schema inference
2. **Data Merging**: Join CVE, products, vendor, and vulnerability data
3. **Text Preprocessing**: TensorFlow tokenization and sequence padding
4. **Label Encoding**: Manual categorical encoding using TensorFlow operations
5. **Data Splitting**: Custom train/test split with NumPy
6. **Model Training**: GPU-accelerated deep learning with callbacks

## 🔧 Requirements

### System Requirements
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM (for large datasets)
- 50GB+ free storage

### Dependencies
```
tensorflow>=2.14.0
pyspark>=3.5.0
numpy>=1.24.0
```

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd deep-learning-vulnerability-scanner
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install tensorflow pyspark numpy
```

4. **Verify GPU support** (optional but recommended)
```bash
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## 📊 Dataset Structure

Place your CVE dataset files in the `dataset/` directory:
```
dataset/
├── cve.csv           # Main CVE data with summaries and CWE classifications
├── products.csv      # CVE-to-product mappings
├── vendor_product.csv # Vendor-product relationships
└── vendors.csv       # Vendor information
```

### Expected CSV Columns
- **cve.csv**: `cve_id, mod_date, pub_date, cvss, cwe_code, cwe_name, summary, access_*`
- **products.csv**: `cve_id, vulnerable_product`
- **vendor_product.csv**: `vendor, product`
- **vendors.csv**: `vendor`

## 🚀 Usage

### Basic Training
```bash
python tensortrain.py
```

### Advanced Configuration
Modify these parameters in `tensortrain.py`:
```python
# Data sampling (adjust based on dataset size and memory)
merged_df = merged_df.sample(fraction=0.001, seed=42)

# Model hyperparameters
max_words = 10000      # Vocabulary size
max_len = 300          # Sequence length
batch_size = 64        # Training batch size
epochs = 20            # Training epochs

# Memory configuration
.config("spark.driver.memory", "12g")
.config("spark.executor.memory", "12g")
```

### Training Output
The training process generates:
- **Training logs**: Real-time accuracy and loss metrics
- **Model file**: `deep_vuln_model.keras`
- **Tokenizer**: `tokenizer.json`
- **Label mappings**: `label_to_int.txt`

## 📈 Performance Metrics

### Training Results
- **Final Test Accuracy**: 86.9%
- **Training Time**: ~20 epochs (2-3 hours on RTX 3050)
- **GPU Memory Usage**: ~2.1GB VRAM
- **Model Size**: ~50MB

### Training Progress Example
```
Epoch 1/20: accuracy: 0.1452 - val_accuracy: 0.3370
Epoch 10/20: accuracy: 0.7476 - val_accuracy: 0.7620
Epoch 20/20: accuracy: 0.8344 - val_accuracy: 0.8690
```

## 📁 Project Structure

```
deep-learning-vulnerability-scanner/
├── tensortrain.py              # Main training script
├── dataset/                    # Dataset directory
│   ├── cve.csv
│   ├── products.csv
│   ├── vendor_product.csv
│   └── vendors.csv
├── deep_vuln_model.keras       # Trained model (generated)
├── tokenizer.json              # Text tokenizer (generated)
├── label_to_int.txt           # Label mappings (generated)
├── temp_data.parquet/         # Temporary processing files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Configuration Options

### Memory Optimization
For smaller systems or larger datasets:
```python
# Reduce sampling fraction
merged_df = merged_df.sample(fraction=0.0001, seed=42)

# Adjust Spark memory
.config("spark.driver.memory", "8g")
.config("spark.executor.memory", "8g")

# Reduce batch size
batch_size = 32
```

### Model Architecture Tuning
```python
# Adjust LSTM units
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

# Modify dense layers
tf.keras.layers.Dense(256, activation='relu')

# Tune dropout rates
tf.keras.layers.Dropout(0.2)  # Less aggressive dropout
```

## 🐛 Troubleshooting

### Common Issues

**OutOfMemoryError**
- Reduce `fraction` parameter in sampling
- Decrease Spark memory configuration
- Use smaller `chunk_size` for data collection

**GPU Not Detected**
```bash
# Install CUDA toolkit
nvidia-smi  # Verify GPU
pip install tensorflow[and-cuda]  # GPU-enabled TensorFlow
```

**CSV Header Mismatch**
- Check CSV files have proper headers
- Verify column names match expected structure
- Handle unnamed columns (like `_c0`)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **Apache Spark Team** for distributed data processing
- **CVE/CWE Communities** for vulnerability data standards
- **NVIDIA** for GPU acceleration support

***

**Built with ❤️ using TensorFlow and Apache Spark**

For questions or support, please open an issue in the repository.
