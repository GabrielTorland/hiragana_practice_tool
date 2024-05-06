# Hiragana Practice Tool

A Python-based application designed to help users practice and improve their recognition of Hiragana characters through a simple and interactive interface.

# Features
- Interactive GUI for practicing Hiragana character recognition
- Model evaluation and training notebooks included.
- Visual feedback using checkmarks and crosses to indicate correct or incorrect answers.

# Setup
1. Clone the Repository
```sh
git clone https://github.com/<your-github-username>/hiragana_practice_tool.git
cd hiragana_practice_tool
```

2. Install Required Python Packages
```sh
pip install -r requirements.txt
```
3. Run the Application
```sh
python app.py
```
## Usage

After starting the application:
- The web interface will guide you through different Hiragana characters.
- Submit your answer to see if it was correct.
- Use the evaluation notebook (`eval_tm.ipynb`) to test the model's accuracy.
- Use the training notebook (`train.ipynb`) to retrain the model using new data.

## Files and Directories

- `app.py`: The main Python file that runs the web server.
- `characters.py`, `models.py`, `utils.py`: Helper scripts used by the main application.
- `config.py`: Configuration settings for the application.
- `model.pk1`: Serialized file of the trained model.
- `k49_classmap.csv`: CSV file mapping labels to characters.
- `static/`: Directory containing static files like images (check marks and x marks).
- `requirements.txt`: File listing all dependencies needed to run the project.
- `*.ipynb`: Jupyter notebooks for training and evaluating the model.
- `k49-train.py`: Script to train MobileNetV3 model
- `Kuzushijidataset.py`: Class to download and import k49 dataset
- `mobilenet_v3_large_50-epochs_32-resize_0.0001-lr_1-dim_fin.pth`: 50 epoch trained MobileNetV3Large on K49
