import subprocess
import time

def run(cmd):
    print(f"\n🔹 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise Exception(f"Failed: {cmd}")

# STEP 1: Run DVC pipeline
run("dvc repro -f")

# STEP 2: Register model in MLflow
run("python src/register.py")

# STEP 3: Wait a bit (safe sync)
time.sleep(2)

# STEP 4: Start Streamlit app
run("streamlit run app/app.py")