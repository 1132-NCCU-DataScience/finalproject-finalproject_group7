"""
M2 EDA — Global Spatial Autocorrelation (Global Moran's I).

Implements §2.1 of the architecture document. 
This script calculates the Global Moran's I to verify the overall spatial 
clustering tendency of YouBike shortages across the city. It reuses the 
row-normalized k-NN(k=6) weight matrix from the local LISA implementation.

Usage:
    python -m m2.eda_global_moran --static data/youbike_station.csv --snapshot data/latest.csv
"""

import argparse
import sys
import pandas as pd
from esda.moran import Moran

# Import the internal function from the team's existing module 
# to ensure the W matrix logic matches the online inference exactly.
try:
    # Primary attempt: running from the project root
    from m2.lisa import _build_knn_w
except ImportError:
    try:
        # Fallback: running directly inside the m2/ directory
        from lisa import _build_knn_w
    except ImportError:
        raise ImportError(
            "Module 'lisa' not found. Please ensure this script is run in the "
            "correct directory, or add the project root to your PYTHONPATH."
        )

def compute_global_moran(
    snapshot_df: pd.DataFrame, 
    station_static: pd.DataFrame, 
    permutations: int = 999
) -> dict:
    """
    Calculate the Global Moran's I for spatial autocorrelation.

    Args:
        snapshot_df (pd.DataFrame): Snapshot data containing current shortage rates. 
                                    The index must be 'sno', and it must contain a 'shortage_rate' column.
        station_static (pd.DataFrame): Static station coordinates containing 'sno', 'latitude', and 'longitude'.
        permutations (int, optional): Number of conditional permutations for the pseudo p-value. Defaults to 999.

    Returns:
        dict: A dictionary containing Morans_I, Expected_I, p_value, and z_score.
    """
    
    # 1. Build the row-normalized K-NN spatial weight matrix
    w = _build_knn_w(snapshot_df, station_static)
    
    # 2. Extract the target variable
    if "shortage_rate" not in snapshot_df.columns:
        raise ValueError("snapshot_df must contain a 'shortage_rate' column.")
        
    x = snapshot_df["shortage_rate"].to_numpy(dtype=float)
    
    # 3. Perform the Global Moran's I statistical test
    moran_global = Moran(x, w, permutations=permutations)
    
    # 4. Return results rounded to 4 decimal places
    return {
        "Morans_I": round(moran_global.I, 4),
        "Expected_I": round(moran_global.EI, 4),
        "p_value": round(moran_global.p_sim, 4),
        "z_score": round(moran_global.z_sim, 4),
    }

def _main():
    """CLI Entrypoint"""
    parser = argparse.ArgumentParser(
        description="Calculate Global Moran's I for YouBike shortage rates."
    )
    parser.add_argument(
        "--static", 
        type=str, 
        default="youbike_station.csv", 
        help="Path to the static station coordinates CSV file."
    )
    parser.add_argument(
        "--snapshot", 
        type=str, 
        required=True, 
        help="Path to the snapshot CSV file containing current shortage rates."
    )
    parser.add_argument(
        "--permutations", 
        type=int, 
        default=999, 
        help="Number of permutations for significance testing (default: 999)."
    )
    
    args = parser.parse_args()

    # Load data and perform basic validation
    try:
        station_static = pd.read_csv(args.static)
        snapshot_df = pd.read_csv(args.snapshot)
        
        # Ensure 'sno' is set as the index and cast to string (matching lisa.py behavior)
        snapshot_df["sno"] = snapshot_df["sno"].astype(str)
        snapshot_df = snapshot_df.set_index("sno")
        station_static["sno"] = station_static["sno"].astype(str)
        
    except FileNotFoundError as e:
        print(f"[Error] Failed to load file: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"[Error] Missing required column (ensure 'sno' exists): {e}")
        sys.exit(1)

    print(f"Calculating Global Moran's I (with {args.permutations} permutations)...")
    
    try:
        results = compute_global_moran(
            snapshot_df, 
            station_static, 
            permutations=args.permutations
        )
        
        print("\n" + "="*50)
        print("🌍 Global Spatial Autocorrelation (Moran's I) Results")
        print("="*50)
        print(f"Observed Moran's I : {results['Morans_I']}")
        print(f"Expected I         : {results['Expected_I']}")
        print(f"Pseudo p-value     : {results['p_value']}")
        print(f"Z-score            : {results['z_score']}")
        print("="*50)
        
        if results['p_value'] < 0.05 and results['Morans_I'] > 0:
            print("\n[Conclusion] p-value < 0.05. The null hypothesis is rejected.")
            print("The YouBike shortage rates exhibit SIGNIFICANT POSITIVE spatial clustering.")
            print("This justifies the inclusion of spatial lag features in the ML pipeline.")
        else:
            print("\n[Conclusion] Results are not statistically significant, or no positive clustering found.")
            print("The current shortage distribution is close to random.")
            
    except Exception as e:
        print(f"[Error] An error occurred during computation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    _main()
