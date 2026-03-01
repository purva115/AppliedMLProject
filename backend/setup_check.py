"""
Quick setup script to verify installation and download NLTK data
Run this after installing requirements.txt
"""
import sys

def check_imports():
    """Check if all required packages can be imported"""
    print("🔍 Checking package imports...")
    
    required_packages = [
        ('flask', 'Flask'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('nltk', 'NLTK'),
        ('googletrans', 'GoogleTrans'),
        ('gtts', 'gTTS'),
        ('speech_recognition', 'SpeechRecognition'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]
    
    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📦 Downloading NLTK data...")
    import nltk
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=False)
            print(f"  ✅ {dataset}")
        except Exception as e:
            print(f"  ❌ {dataset} - {e}")
    
    print("\n✅ NLTK data downloaded!")

def check_dataset():
    """Check if dataset exists"""
    print("\n📊 Checking for dataset...")
    import os
    
    dataset_path = '../data/raw/medicine_dataset.csv'
    
    if os.path.exists(dataset_path):
        print(f"  ✅ Dataset found at {dataset_path}")
        
        # Check file size
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"  📁 File size: {size_mb:.2f} MB")
        
        return True
    else:
        print(f"  ❌ Dataset NOT found at {dataset_path}")
        print("\n  📥 Please download the Kaggle medicine dataset:")
        print("     1. Go to: https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details")
        print("     2. Download and place as: data/raw/medicine_dataset.csv")
        print("     3. Then run: python ../data/prepare_dataset.py")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("🏥 PharmaLLM Backend Setup Verification")
    print("=" * 60)
    
    # Check imports
    if not check_imports():
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Check dataset
    dataset_ready = check_dataset()
    
    print("\n" + "=" * 60)
    if dataset_ready:
        print("✅ Setup complete! Ready to run the backend.")
        print("\nNext steps:")
        print("  1. Run backend: python app.py")
        print("  2. Test at: http://localhost:5000/api/health")
    else:
        print("⚠️  Setup incomplete - dataset needed.")
        print("\nNext steps:")
        print("  1. Download the Kaggle dataset (see instructions above)")
        print("  2. Preprocess: python ../data/prepare_dataset.py")
        print("  3. Run backend: python app.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
