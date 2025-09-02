# house-price-predictor

This project predicts house prices using machine learning models.  
It takes housing features (like size, year built, and location) and outputs a predicted sale price.

# How to Run

# 1. Clone the repository
    git clone https://github.com/regitting/house-price-predictor.git
    cd house-price-predictor


# 2. Create and activate a virtual environment:
    py -m venv .venv
    .\.venv\Scripts\activate.bat   # on Windows (cmd)
    .\.venv\Scripts\Activate.ps1   # on Windows (PowerShell)
    source .venv/bin/activate      # on Mac/Linux

# 3. Install the required packages:
    pip install -r requirements.txt

# 4. Run training/predicting software one at a time:
    Demo mode (generates 1000 rows of California housing data)
    python src\data_loader.py --demo --rows 1000
    python src\train.py
    python src\predict.py --csv data\california_example.csv

    OR, with your own CSV:
    python src\train.py
    python src\predict.py --csv data\your_file.csv

# 5. See results:
    Run show_predictions.ipynb notebook to see results

# NOTE:
- The demo only uses 1000 rows from California.
- For real use, provide your own CSV with the expected columns listed in `models/metadata.json`.
- Extra columns will be dropped automatically; missing columns will be added as NaN (handled by the preprocessing pipeline).
- Predictions will be saved to the `predictions/` folder with a timestamped filename.

The script will train a model and print the evaluation results. A trained model file will be saved in the models/ folder.