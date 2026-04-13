"""
Setup script for Knowledge Graph with XGBoost
This script helps set up the environment and provides examples
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Core packages
    packages = [
        "pyspark==3.4.0",
        "xgboost==1.7.5", 
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.1",
        "seaborn==0.12.2"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    # GraphFrames needs special handling
    print("\nNote: GraphFrames will be downloaded automatically when running PySpark")
    print("Make sure you have Java 8+ installed for PySpark to work properly")

def check_java():
    """Check if Java is installed"""
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Java is installed")
            return True
        else:
            print("✗ Java is not installed or not in PATH")
            return False
    except FileNotFoundError:
        print("✗ Java is not installed or not in PATH")
        return False

def setup_environment():
    """Setup environment variables for PySpark"""
    print("Setting up environment...")
    
    # Set PYSPARK_PYTHON to current Python executable
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    print("✓ Environment variables set")

if __name__ == "__main__":
    print("=" * 50)
    print("KNOWLEDGE GRAPH + XGBOOST SETUP")
    print("=" * 50)
    
    # Check Java
    java_ok = check_java()
    if not java_ok:
        print("\nPlease install Java 8+ before proceeding:")
        print("- Windows: Download from https://adoptium.net/")
        print("- macOS: brew install openjdk@11")
        print("- Linux: sudo apt-get install openjdk-11-jdk")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Install packages
    install_requirements()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("You can now run: python knowledge_graph_xgboost.py")
