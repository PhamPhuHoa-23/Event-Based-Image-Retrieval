#!/usr/bin/env python3
"""
Monitor script to track upload progress
"""

import requests
import time
import datetime

def check_count():
    try:
        response = requests.get("http://localhost:9200/articles/_count")
        if response.status_code == 200:
            return response.json()["count"]
    except:
        return None

def main():
    print("🔍 MONITORING UPLOAD PROGRESS")
    print("=" * 40)
    
    prev_count = 0
    start_time = time.time()
    
    while True:
        current_count = check_count()
        if current_count is not None:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            rate = current_count - prev_count
            
            elapsed = time.time() - start_time
            if elapsed > 0:
                avg_rate = current_count / elapsed
                print(f"[{current_time}] Articles: {current_count:,} (+{rate:,}) | Avg: {avg_rate:.1f}/sec")
            
            prev_count = current_count
        else:
            print("❌ Failed to get count")
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    main() 