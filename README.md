# ✍️ Signature Verification AI

A web-based signature verification system using deep learning to determine if two signatures belong to the same person.

## 🚀 Live Demo

🌐 **[Try it live on Streamlit Cloud!](https://your-app-name.streamlit.app)**

*Deploy your own version in minutes following the guide below*

## 🎯 Features

### Single Signature Comparison
- **Upload & Compare**: Simply upload two signature images
- **Real-time Analysis**: Get instant similarity scores
- **Adjustable Threshold**: Customize sensitivity settings
- **Visual Results**: Clear confidence metrics and match indicators

### Batch Processing 🆕
- **Multiple Verification**: Compare many signatures against a reference
- **Bulk Upload**: Process dozens of signatures at once
- **CSV Export**: Download detailed results for analysis
- **Summary Dashboard**: Visual charts and statistics

### Advanced Features
- **High Accuracy**: 98.75% validation accuracy on test data
- **Custom Thresholds**: Adjust sensitivity for different use cases
- **Progress Tracking**: Real-time processing updates
- **Error Handling**: Graceful failure management

## 🧠 How It Works

This application uses a **Siamese Neural Network** architecture:

1. **Feature Extraction**: Convolutional layers extract signature characteristics
2. **Similarity Comparison**: Twin networks process both signatures simultaneously
3. **Distance Calculation**: Measures feature similarity between signatures
4. **Classification**: Outputs probability that signatures match

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: TensorFlow/Keras deep learning
- **Model**: Siamese CNN with 98.75% accuracy
- **Deployment**: Streamlit Cloud / Heroku ready

## 📦 Installation & Setup

### Option 1: Run Locally

```bash
# Clone the repository
git clone [your-repo-url]
cd signature-verification-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

### Option 2: Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy directly from your fork

## 📊 Model Performance

- **Training Accuracy**: 96.75%
- **Validation Accuracy**: 98.75%
- **Architecture**: Siamese CNN
- **Input Size**: 100x100 grayscale images
- **Dataset**: Handwritten signatures from multiple writers

## 🎮 Usage

1. **Upload Images**: Choose two signature images (PNG, JPG, JPEG)
2. **Click Verify**: Press the verification button
3. **View Results**: Get similarity score and match confidence
4. **Interpret**: Green = match, Red = no match, Yellow = uncertain

## 📝 Tips for Best Results

- Use clear, high-contrast signature images
- Ensure signatures are properly cropped
- Avoid blurry or low-quality images
- Test with both genuine and forged signatures

## 🔧 Technical Details

### Model Architecture
```
Input Layer (100x100x1)
├── Conv2D (32 filters, 3x3)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3)
├── MaxPooling2D (2x2)
├── Flatten
└── Dense (128 units)

Siamese Network
├── Feature Extraction (Base CNN)
├── Distance Calculation (L1 distance)
└── Classification (Sigmoid output)
```

### Training Process
- **Dataset**: Kaggle handwritten signatures dataset
- **Epochs**: 10
- **Batch Size**: 16
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
- Free hosting for public repos
- Automatic updates from GitHub
- Easy setup and management

### Heroku
- Add `setup.sh` and `Procfile` for Heroku deployment
- Supports custom domains
- Scalable hosting options

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [Handwritten Signatures Dataset](https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures)
- Framework: TensorFlow/Keras for deep learning
- Interface: Streamlit for web application

## 📞 Contact

- GitHub: [https://github.com/HabibaAmrhu]
- Email: [habibaamr360@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/habiba360?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B60mPuyBNSNmPUIjZ9n%2BrqA%3D%3D]

---

