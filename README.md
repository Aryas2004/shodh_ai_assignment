# Loan Approval Prediction — Deep Learning vs Offline Reinforcement Learning (CQL)

This project compares **Deep Learning (DL)** and **Offline Reinforcement Learning (RL)** methods for a loan approval problem using the LendingClub dataset (or a similar financial dataset).  
It includes data preprocessing, deep learning model training, RL dataset creation, and offline RL training using **Conservative Q-Learning (CQL)**.

---

## ⚙️ Setup Instructions

### 1️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows PowerShell
2️⃣ Install dependencies
All dependencies are listed in requirements.txt.

bash
Copy code
pip install -r requirements.txt
If you have a GPU, install the correct PyTorch version for your CUDA version (see PyTorch Installation Guide).

🧠 How to Run the Project
Step 1: Data Preparation
Run:

bash
Copy code
python preparationData.py
This script:

Loads and cleans the raw dataset.

Encodes categorical variables.

Splits into training, validation, and test sets.

Saves processed arrays (.npy) and preprocessing artifacts (like a scaler or encoder) for later use.

Expected outputs:
Preprocessed NumPy arrays and preprocessing objects saved to a local folder (e.g., models/ or similar, depending on your script paths).

Step 2: Prepare the Offline RL Dataset
Run:

bash
Copy code
python prepare_rl_dataset.py
This script:

Uses the preprocessed data from Step 1.

Converts it into a Reinforcement Learning–compatible dataset of (state, action, reward, next state) tuples.

Saves the dataset to a serialized file (e.g., .pkl or .pt).

Expected outputs:
Offline RL dataset files (for example: rl_loan_dataset.pkl, rl_scaler.pkl).

Step 3: Train the Deep Learning Model
Run:

bash
Copy code
python deepLearning_train.py
This script:

Trains a supervised neural network classifier on the prepared dataset.

Uses validation data to tune hyperparameters.

Saves the best-performing model checkpoint.

Expected outputs:
A trained deep learning model file (for example: best_mlp.pth or model.pth).

Step 4: Train the Offline RL Agent (CQL)
Run:

bash
Copy code
python reinforce_train.py
This script:

Loads the offline RL dataset.

Initializes and trains a Conservative Q-Learning (CQL) agent.

Periodically evaluates policy performance.

Saves the final trained RL agent.

Expected outputs:
CQL model checkpoint (for example: cql_loan_agent.pth) and optional training plots.

Step 5: Compare Model Performances
Run:

bash
Copy code
python comparision_models.py
This script:

Loads the trained DL and RL models.

Evaluates both models on the test dataset.

Generates metrics and plots comparing performance.

Expected outputs:
Comparison results (printed metrics and possibly figures such as accuracy or reward curves).

Step 6: Verify Environment and Setup
Run:

bash
Copy code
python reinforce_int.py
This performs:

Environment validation (checks required files and dependencies).

Small sanity checks on dataset and models.

Prints confirmation that all steps can run successfully.

🧾 Example Command Sequence
bash
Copy code
python preparationData.py
python prepare_rl_dataset.py
python deepLearning_train.py
python reinforce_train.py
python comparision_models.py
python reinforce_int.py
☁️ Running in Google Colab
If you’re running this in Google Colab:

Mount your Google Drive:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/<your-folder>
Install dependencies:

bash
Copy code
!pip install -r requirements.txt
Upload your dataset into /content/data/ or update the paths in preparationData.py and prepare_rl_dataset.py.

Then run the same command sequence above, replacing python with !python.

✅ Tip: In Colab, enable GPU under Runtime → Change runtime type → GPU for faster training.
