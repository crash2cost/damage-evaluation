#!/usr/bin/env python3
"""
Cost Estimation Model Training - Car Repair Cost Prediction
============================================================
Uses scikit-learn for regression modeling.
Compares RandomForest vs GradientBoosting and saves the best model.

Features:
- Part_Name: Which car part is damaged
- Severity: Damage severity level (1-5)
- Car_Segment: Vehicle price segment

Usage:
    python train.py                                  # Train with defaults
    python train.py --data-path path/to/data.csv    # Custom data
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "cost_model" / "dataset"
MODEL_DIR = ROOT / "cost_model" / "models"

MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    """Training configuration."""
    data_path: Path = DATA_DIR / "detailed_repair_costs.csv"
    random_state: int = 42
    test_size: float = 0.3
    # Random Forest parameters
    rf_estimators: int = 200
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    # Gradient Boosting parameters
    gb_estimators: int = 200
    gb_max_depth: int = 7
    gb_learning_rate: float = 0.1
    gb_subsample: float = 0.8


FEATURE_COLUMNS = ["Part_Encoded", "Severity", "Segment_Encoded"]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate the dataset."""
    df = pd.read_csv(data_path)
    
    required = {"Part_Name", "Severity", "Car_Segment", "Estimated_Cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features with label encoding."""
    part_encoder = LabelEncoder()
    segment_encoder = LabelEncoder()
    
    df = df.copy()
    df["Part_Encoded"] = part_encoder.fit_transform(df["Part_Name"])
    df["Segment_Encoded"] = segment_encoder.fit_transform(df["Car_Segment"])
    
    X = df[FEATURE_COLUMNS]
    y = df["Estimated_Cost"]
    
    return X, y, part_encoder, segment_encoder


def evaluate_model(model, X, y, set_name: str) -> dict:
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n{set_name} Set:")
    print(f"  MAE:  ₪{mae:.2f}")
    print(f"  RMSE: ₪{rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train(config: TrainingConfig):
    """
    Train cost estimation models.
    
    Trains both RandomForest and GradientBoosting, saves the best one.
    """
    print(f"\n{'='*60}")
    print("💰 Cost Estimation Model Training")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data from: {config.data_path}")
    df = load_data(config.data_path)
    print(f"Total samples: {len(df)}")
    
    # Show data statistics
    print(f"\nData Statistics:")
    print(df.describe())
    
    # Prepare features
    X, y, part_encoder, segment_encoder = prepare_features(df)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.random_state
    )
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # Train Random Forest
    print(f"\n{'='*60}")
    print("Training Random Forest...")
    print(f"{'='*60}")
    
    rf_model = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
        verbose=1,
    )
    rf_model.fit(X_train, y_train)
    
    rf_train = evaluate_model(rf_model, X_train, y_train, "Train")
    rf_val = evaluate_model(rf_model, X_val, y_val, "Validation")
    rf_test = evaluate_model(rf_model, X_test, y_test, "Test")
    
    # Train Gradient Boosting
    print(f"\n{'='*60}")
    print("Training Gradient Boosting...")
    print(f"{'='*60}")
    
    gb_model = GradientBoostingRegressor(
        n_estimators=config.gb_estimators,
        max_depth=config.gb_max_depth,
        learning_rate=config.gb_learning_rate,
        subsample=config.gb_subsample,
        random_state=config.random_state,
        verbose=1,
    )
    gb_model.fit(X_train, y_train)
    
    gb_train = evaluate_model(gb_model, X_train, y_train, "Train")
    gb_val = evaluate_model(gb_model, X_val, y_val, "Validation")
    gb_test = evaluate_model(gb_model, X_test, y_test, "Test")
    
    # Select best model based on validation MAE
    if rf_val["mae"] < gb_val["mae"]:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_test
        other_model = gb_model
        other_name = "gradient_boosting"
    else:
        best_model = gb_model
        best_name = "Gradient Boosting"
        best_metrics = gb_test
        other_model = rf_model
        other_name = "random_forest"
    
    print(f"\n{'='*60}")
    print(f"✅ Best Model: {best_name}")
    print(f"   Test MAE: ₪{best_metrics['mae']:.2f}")
    print(f"   Test R²:  {best_metrics['r2']:.4f}")
    print(f"{'='*60}")
    
    # Save models and encoders
    joblib.dump(best_model, MODEL_DIR / "cost_estimator.pkl")
    joblib.dump(other_model, MODEL_DIR / f"cost_estimator_{other_name}.pkl")
    joblib.dump(part_encoder, MODEL_DIR / "part_encoder.pkl")
    joblib.dump(segment_encoder, MODEL_DIR / "segment_encoder.pkl")
    
    # Save metadata
    metadata = {
        "best_model": best_name,
        "test_metrics": best_metrics,
        "parts": part_encoder.classes_.tolist(),
        "segments": segment_encoder.classes_.tolist(),
        "severity_range": [int(df["Severity"].min()), int(df["Severity"].max())],
        "feature_names": FEATURE_COLUMNS,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Models saved to: {MODEL_DIR}")
    
    # Show example predictions
    print(f"\n{'='*60}")
    print("Example Predictions:")
    print(f"{'='*60}")
    
    examples = [
        ("Front Bumper", 3, "Family"),
        ("Headlight", 5, "Luxury"),
        ("Front Door", 2, "Micro"),
        ("Hood", 4, "SUV"),
    ]
    
    for part, severity, segment in examples:
        try:
            part_enc = part_encoder.transform([part])[0]
            seg_enc = segment_encoder.transform([segment])[0]
            cost = best_model.predict([[part_enc, severity, seg_enc]])[0]
            print(f"{part} (Severity {severity}, {segment}): ₪{cost:,.0f}")
        except ValueError:
            print(f"{part}: Not in training data")
    
    return best_model


def predict(part_name: str, severity: int, car_segment: str):
    """Predict repair cost for given inputs."""
    model = joblib.load(MODEL_DIR / "cost_estimator.pkl")
    part_encoder = joblib.load(MODEL_DIR / "part_encoder.pkl")
    segment_encoder = joblib.load(MODEL_DIR / "segment_encoder.pkl")
    
    part_enc = part_encoder.transform([part_name])[0]
    seg_enc = segment_encoder.transform([car_segment])[0]
    
    cost = model.predict([[part_enc, severity, seg_enc]])[0]
    
    print(f"Part: {part_name}")
    print(f"Severity: {severity}/5")
    print(f"Car Segment: {car_segment}")
    print(f"Estimated Cost: ₪{cost:,.0f}")
    
    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost Estimation Training")
    parser.add_argument("--data-path", type=Path, default=DATA_DIR / "detailed_repair_costs.csv",
                       help="Path to training data CSV")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_path=args.data_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    
    train(config)
