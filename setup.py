#!/usr/bin/env python3
"""
Setup script for Healthcare Intake Assistant
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has the required API key"""
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        print("Please create a .env file in the root directory with:")
        print("ELEVEN_LABS_API_KEY=your_api_key_here")
        return False
    
    with open(".env", "r") as f:
        content = f.read()
        if "ELEVEN_LABS_API_KEY" in content:
            print("✅ .env file found with ELEVEN_LABS_API_KEY")
            return True
        else:
            print("❌ ELEVEN_LABS_API_KEY not found in .env file")
            print("Please add: ELEVEN_LABS_API_KEY=your_api_key_here")
            return False

def main():
    print("🏥 Healthcare Intake Assistant Setup")
    print("=" * 40)
    
    # Check environment file
    env_ok = check_env_file()
    
    # Install requirements
    install_ok = install_requirements()
    
    if env_ok and install_ok:
        print("\n✅ Setup complete!")
        print("🚀 Run the transcriber with: python src/transcribe_speech.py")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")

if __name__ == "__main__":
    main()
