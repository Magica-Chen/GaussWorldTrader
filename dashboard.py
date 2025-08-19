#!/usr/bin/env python3
"""
Gauss World Trader - Dashboard Launcher
Runs Streamlit dashboard on port 3721
"""

import sys
import os
import subprocess

def main():
    print("🌍 Starting Gauss World Trader Dashboard on port 3721")
    print("📱 Open your browser to: http://localhost:3721")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        # Add the project root to the Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Run streamlit with custom port
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/simple_dashboard.py",  # Updated path to ui directory
            "--server.port", "3721",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--server.fileWatcherType", "none"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Make sure src/ui/simple_dashboard.py exists and Streamlit is installed")

if __name__ == '__main__':
    main()