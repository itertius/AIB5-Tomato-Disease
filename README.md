# 🍅 AI Builders 5 - Tomato Disease Classification

**AIB5-Tomato-Disease** is a computer vision project focused on detecting and classifying diseases in tomato plants using deep learning. The model is trained on image datasets and designed for potential deployment on real-world devices.

## 📂 Project Structure

```
AIB5-Tomato-Disease/
├── data/                # Data directory (datasets, splits)
├── model/               # Trained model weights
├── notebooks/           # Jupyter notebooks organized by week
├── requirements.txt     # Python package dependencies
├── app.py               # Streamlit deployment app
└── README.md            # Project documentation
```

## 🧠 Model Details

- **Architecture**: MobileNetV2  
- **Framework**: PyTorch  
- **Input Size**: 224x224 RGB  
- **Loss Function**: CrossEntropyLoss  
- **Optimizer**: Adam  
- **Dataset**: PlantVillage (Tomato subset with 10 classes)

## 🔧 Installation

```bash
git clone https://github.com/itertius/AIB5-Tomato-Disease.git
cd AIB5-Tomato-Disease
pip install -r requirements.txt
```

## 🚀 Run Streamlit App Locally

To launch the web app locally for disease classification using your webcam or image upload:
1. Ensure you are in the project root directory (`AIB5-Tomato-Disease`)
2. Run the Streamlit app with:
```
bash
streamlit run app.py
```

3. This will open a local web server (usually at `http://localhost:8501`) in your browser where you can interact with the app.

## 🙏 Special Thank
- Mentor : **พี่เต๋อ** — นายปัญญาวุธ ศรีอิสรานุสรณ์ (Panyawut Sri-iesaranusorn)
- TA : **พี่โชกุน** — นายเสฎฐพันธ์ เหล่าอารีย์ (Settapun Laoaree)
- เพื่อนๆ ในโครงการ AI Builders

## 🔗 Reference
- [AI Builders Curriculum](https://github.com/ai-builders/curriculum)  
- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- [MobileNetV2 Paper and Code](https://paperswithcode.com/method/mobilenetv2)

## 🌐 Link
- Deployment : https://itertius-aib5-tomato-disease.streamlit.app/  
- Github : https://github.com/itertius/AIB5-Tomato-Disease/  
- Dataset : https://www.kaggle.com/datasets/emmarex/plantdisease/

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
