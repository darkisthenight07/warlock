# End-to-End RL Based Strategy in Cryptocurrency Markets

**Authors:**  
Jalaj Bhadouria  
Jay Gupta  
Mandar Bhalerao

### Installation & Execution Steps

Run the following commands sequentially from your repository root (`warlock-main`):

```powershell
# 1. Create a virtual environment named 'venv'
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Download and clean data (Outputs to data/processed/*.parquet)
python runner.py

# 5. Generate features, perform dataset splitting, and produce diagnostic graphs
python data_analysis.py
