#!/usr/bin/env python3
"""
Dashboard launcher script with custom port
"""

import os
import sys
import subprocess

def run_dashboard(port=3721):
    """Run the dashboard on specified port"""
    print(f"🚀 Starting Trading System Dashboard on port {port}")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        # Run streamlit with custom port
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "simple_dashboard.py", 
            "--server.port", str(port),
            "--server.address", "localhost",
            "--server.headless", "true",
            "--server.fileWatcherType", "none"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    run_dashboard(3721)