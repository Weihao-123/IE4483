#!/usr/bin/env python3
"""
Sort submission.csv by ID in ascending order (1 to 500)
"""

import pandas as pd

def sort_submission():
    """Sort submission.csv by ID from 1 to 500"""
    print("Sorting submission.csv by ID...")
    
    try:
        # Read the current submission file
        df = pd.read_csv('submission.csv')
        
        print(f"Current file has {len(df)} rows")
        print(f"ID range: {df['ID'].min()} to {df['ID'].max()}")
        
        # Sort by ID
        df_sorted = df.sort_values('ID').reset_index(drop=True)
        
        # Save the sorted file
        df_sorted.to_csv('submission.csv', index=False)
        
        print("✓ Submission file sorted successfully!")
        print(f"✓ File now contains {len(df_sorted)} predictions in order 1-500")
        
        # Show first few rows to verify
        print("\nFirst 10 rows:")
        print(df_sorted.head(10))
        
        # Show last few rows to verify
        print("\nLast 10 rows:")
        print(df_sorted.tail(10))
        
    except Exception as e:
        print(f"Error sorting submission file: {e}")

if __name__ == "__main__":
    sort_submission()


