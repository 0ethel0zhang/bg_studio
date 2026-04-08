import subprocess
import os

def test_sips_conversion(heic_path):
    jpg_path = "test_sips.jpg"
    if os.path.exists(jpg_path):
        os.remove(jpg_path)
    
    print(f"Testing sips conversion for {heic_path}...")
    try:
        result = subprocess.run(["sips", "-s", "format", "jpeg", heic_path, "--out", jpg_path], capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(jpg_path):
            print(f"  Successfully converted {heic_path} to {jpg_path}")
            os.remove(jpg_path)
            return True
        else:
            print(f"  Failed to convert {heic_path}. Return code: {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Exception during conversion: {e}")
        return False

if __name__ == "__main__":
    heic_files = [f for f in os.listdir('.') if f.lower().endswith('.heic')]
    if not heic_files:
        print("No HEIC files found.")
    else:
        test_sips_conversion(heic_files[0])
