# house-price-predictor

This project predicts house prices using machine learning models.  
It takes housing features (like size, year built, and location) and outputs a predicted sale price.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/regitting/house-price-predictor.git
   cd house-price-predictor

2. Create and activate a virtual environment:
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1   # on Windows
    source .venv/bin/activate      # on Mac/Linux

3. Install the required packages:
    pip install -r requirements.txt

4. Run the training script:
    python src/train.py

The script will train a model and print the evaluation results. A trained model file will be saved in the models/ folder.