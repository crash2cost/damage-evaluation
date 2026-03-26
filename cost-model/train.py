#!/usr/bin/env python3
"""
Cost Estimation Model Training - Car Repair Cost Prediction (v3)
=================================================================
Improved model that uses:
- Part_Name: Which car part is damaged (Front Bumper, Hood, etc.)
- Damage_Type: Type of damage (bumper_dent, door_scratch, etc.)
- Severity: Damage severity level (1-5)
- Car_Category: Vehicle category affecting repair cost (sedan, luxury, etc.)
- Interaction features: Severity², Part×Severity, Damage×Severity

Usage:
    python train.py                                  # Train with defaults
    python train.py --data-path path/to/data.csv    # Custom data
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Constants
# =============================================================================

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "cost-model" / "dataset"
MODEL_DIR = ROOT / "cost-model" / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Training defaults
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

# Random Forest defaults
DEFAULT_RF_ESTIMATORS = 500
DEFAULT_RF_MAX_DEPTH = 20
DEFAULT_RF_MIN_SAMPLES_SPLIT = 3
DEFAULT_RF_MIN_SAMPLES_LEAF = 1

# Gradient Boosting defaults
DEFAULT_GB_ESTIMATORS = 500
DEFAULT_GB_MAX_DEPTH = 8
DEFAULT_GB_LEARNING_RATE = 0.05
DEFAULT_GB_SUBSAMPLE = 0.8

# Cross-validation
CV_FOLDS = 5
CV_STD_MULTIPLIER = 2

# Display
SEPARATOR_WIDTH = 60
PERCENTAGE_MULTIPLIER = 100

# Severity
MAX_SEVERITY = 5

# Model version
MODEL_VERSION = 3

# Mapping from API car_segment names to dataset Car_Category names
SEGMENT_TO_CATEGORY = {
    "Small": "small",
    "Sedan": "sedan",
    "Family": "family_suv",
    "SUV": "family_suv",
    "Truck": "truck",
    "Minivan": "minivan",
    "Sports": "sports",
    "Luxury": "luxury",
    "Electric": "electric",
}
DEFAULT_CATEGORY = "sedan"


@dataclass
class TrainingConfig:
    """Training configuration for cost estimation models."""
    data_path: Path = DATA_DIR / "repair_costs_v2.csv"
    random_state: int = DEFAULT_RANDOM_STATE
    test_size: float = DEFAULT_TEST_SIZE
    rf_estimators: int = DEFAULT_RF_ESTIMATORS
    rf_max_depth: int = DEFAULT_RF_MAX_DEPTH
    rf_min_samples_split: int = DEFAULT_RF_MIN_SAMPLES_SPLIT
    rf_min_samples_leaf: int = DEFAULT_RF_MIN_SAMPLES_LEAF
    gb_estimators: int = DEFAULT_GB_ESTIMATORS
    gb_max_depth: int = DEFAULT_GB_MAX_DEPTH
    gb_learning_rate: float = DEFAULT_GB_LEARNING_RATE
    gb_subsample: float = DEFAULT_GB_SUBSAMPLE


# Base feature columns (before interaction features)
BASE_FEATURE_COLUMNS: List[str] = ["Part_Encoded", "Damage_Encoded", "Severity", "Category_Encoded"]

# All feature columns including engineered features
FEATURE_COLUMNS: List[str] = [
    "Part_Encoded", "Damage_Encoded", "Severity", "Category_Encoded",
    "Severity_Sq", "Part_Severity", "Damage_Severity",
]

# Required columns in training data
REQUIRED_COLUMNS = {"Part_Name", "Damage_Type", "Severity", "Estimated_Cost"}


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate the dataset."""
    df = pd.read_csv(data_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # If Car_Category is missing, default to 'sedan'
    if "Car_Category" not in df.columns:
        df["Car_Category"] = DEFAULT_CATEGORY

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered interaction features."""
    df = df.copy()
    df["Severity_Sq"] = df["Severity"] ** 2
    df["Part_Severity"] = df["Part_Encoded"] * df["Severity"]
    df["Damage_Severity"] = df["Damage_Encoded"] * df["Severity"]
    return df


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, LabelEncoder, LabelEncoder]:
    """Prepare features with label encoding and interaction features."""
    part_encoder = LabelEncoder()
    damage_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    df = df.copy()
    df["Part_Encoded"] = part_encoder.fit_transform(df["Part_Name"])
    df["Damage_Encoded"] = damage_encoder.fit_transform(df["Damage_Type"])
    df["Category_Encoded"] = category_encoder.fit_transform(df["Car_Category"])

    df = add_interaction_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["Estimated_Cost"]

    return X, y, part_encoder, damage_encoder, category_encoder


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str
) -> Dict[str, float]:
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * PERCENTAGE_MULTIPLIER

    print(f"\n{set_name} Set:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train(config: TrainingConfig) -> Any:
    """
    Train cost estimation models.

    Trains both RandomForest and GradientBoosting regressors,
    evaluates their performance, and saves the best one.
    """
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Cost Estimation Model Training (v3)")
    print(f"{'=' * SEPARATOR_WIDTH}")

    # Load data
    print(f"\nLoading data from: {config.data_path}")
    df = load_data(config.data_path)
    print(f"Total samples: {len(df)}")

    # Show data statistics
    print("\nData Statistics:")
    print(f"  Parts: {df['Part_Name'].nunique()} unique")
    print(f"  Damage Types: {df['Damage_Type'].nunique()} unique")
    print(f"  Car Categories: {df['Car_Category'].nunique()} unique")
    print(f"  Severity Range: {df['Severity'].min()}-{df['Severity'].max()}")
    print(f"  Cost Range: ${df['Estimated_Cost'].min():,} - ${df['Estimated_Cost'].max():,}")
    print(f"  Mean Cost: ${df['Estimated_Cost'].mean():,.0f}")

    # Prepare features
    X, y, part_encoder, damage_encoder, category_encoder = prepare_features(df)

    print(f"\nFeatures ({len(FEATURE_COLUMNS)}):")
    for col in FEATURE_COLUMNS:
        print(f"  - {col}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    print("\nDataset Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")

    # Train Random Forest
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Training Random Forest...")
    print(f"{'=' * SEPARATOR_WIDTH}")

    rf_model = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Print Random Forest model structure
    print("\nRandom Forest Structure:")
    print(f"  Type: Tree-based ensemble (NOT a neural network)")
    print(f"  Number of trees:      {rf_model.n_estimators}")
    print(f"  Max depth:            {rf_model.max_depth}")
    print(f"  Min samples split:    {rf_model.min_samples_split}")
    print(f"  Min samples leaf:     {rf_model.min_samples_leaf}")
    print(f"  Input features:       {rf_model.n_features_in_}")
    depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
    nodes = [tree.tree_.node_count for tree in rf_model.estimators_]
    leaves = [tree.tree_.n_leaves for tree in rf_model.estimators_]
    print(f"  Avg tree depth:       {np.mean(depths):.1f}")
    print(f"  Avg nodes per tree:   {np.mean(nodes):.0f}")
    print(f"  Avg leaves per tree:  {np.mean(leaves):.0f}")
    print(f"  Total decision nodes: {sum(nodes):,}")

    rf_train = evaluate_model(rf_model, X_train, y_train, "Train")
    rf_test = evaluate_model(rf_model, X_test, y_test, "Test")

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=CV_FOLDS, scoring="neg_mean_absolute_error")
    print(f"\n  CV MAE: ${-cv_scores.mean():.2f} (+/- ${cv_scores.std() * CV_STD_MULTIPLIER:.2f})")

    # Train Gradient Boosting
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Training Gradient Boosting...")
    print(f"{'=' * SEPARATOR_WIDTH}")

    gb_model = GradientBoostingRegressor(
        n_estimators=config.gb_estimators,
        max_depth=config.gb_max_depth,
        learning_rate=config.gb_learning_rate,
        subsample=config.gb_subsample,
        random_state=config.random_state,
    )
    gb_model.fit(X_train, y_train)

    # Print Gradient Boosting model structure
    print("\nGradient Boosting Structure:")
    print(f"  Type: Tree-based boosting ensemble (NOT a neural network)")
    print(f"  Number of estimators: {gb_model.n_estimators}")
    print(f"  Max depth:            {gb_model.max_depth}")
    print(f"  Learning rate:        {gb_model.learning_rate}")
    print(f"  Subsample:            {gb_model.subsample}")
    print(f"  Input features:       {gb_model.n_features_in_}")
    gb_depths = [tree[0].tree_.max_depth for tree in gb_model.estimators_]
    gb_nodes = [tree[0].tree_.node_count for tree in gb_model.estimators_]
    gb_leaves = [tree[0].tree_.n_leaves for tree in gb_model.estimators_]
    print(f"  Avg tree depth:       {np.mean(gb_depths):.1f}")
    print(f"  Avg nodes per tree:   {np.mean(gb_nodes):.0f}")
    print(f"  Avg leaves per tree:  {np.mean(gb_leaves):.0f}")
    print(f"  Total decision nodes: {sum(gb_nodes):,}")

    gb_train = evaluate_model(gb_model, X_train, y_train, "Train")
    gb_test = evaluate_model(gb_model, X_test, y_test, "Test")

    # Cross-validation
    cv_scores = cross_val_score(gb_model, X, y, cv=CV_FOLDS, scoring="neg_mean_absolute_error")
    print(f"\n  CV MAE: ${-cv_scores.mean():.2f} (+/- ${cv_scores.std() * CV_STD_MULTIPLIER:.2f})")

    # Select best model based on test MAE
    if rf_test["mae"] < gb_test["mae"]:
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

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"Best Model: {best_name}")
    print(f"   Test MAE:  ${best_metrics['mae']:.2f}")
    print(f"   Test MAPE: {best_metrics['mape']:.2f}%")
    print(f"   Test R2:   {best_metrics['r2']:.4f}")
    print(f"{'=' * SEPARATOR_WIDTH}")

    # Feature importance
    print("\nFeature Importance:")
    for name, importance in zip(FEATURE_COLUMNS, best_model.feature_importances_):
        print(f"  {name}: {importance:.3f}")

    # Save models and encoders
    joblib.dump(best_model, MODEL_DIR / "cost_estimator.pkl")
    joblib.dump(other_model, MODEL_DIR / f"cost_estimator_{other_name}.pkl")
    joblib.dump(part_encoder, MODEL_DIR / "part_encoder.pkl")
    joblib.dump(damage_encoder, MODEL_DIR / "damage_encoder.pkl")
    joblib.dump(category_encoder, MODEL_DIR / "category_encoder.pkl")

    # Save metadata
    metadata = {
        "version": MODEL_VERSION,
        "best_model": best_name,
        "test_metrics": best_metrics,
        "parts": part_encoder.classes_.tolist(),
        "damage_types": damage_encoder.classes_.tolist(),
        "car_categories": category_encoder.classes_.tolist(),
        "severity_range": [int(df["Severity"].min()), int(df["Severity"].max())],
        "feature_names": FEATURE_COLUMNS,
        "feature_importance": dict(zip(FEATURE_COLUMNS, [float(x) for x in best_model.feature_importances_])),
        "segment_to_category": SEGMENT_TO_CATEGORY,
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to: {MODEL_DIR}")

    # Show example predictions
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Example Predictions:")
    print(f"{'=' * SEPARATOR_WIDTH}")

    examples = [
        ("Front Bumper", "bumper_dent", 3, "sedan"),
        ("Front Door", "door_dent", 4, "luxury"),
        ("Hood", "hood_scratch", 2, "small"),
        ("Windshield", "glass_shatter", 5, "family_suv"),
        ("Headlight", "head_lamp", 3, "sports"),
    ]

    for part, damage_type, severity, category in examples:
        try:
            part_enc = part_encoder.transform([part])[0]
            damage_enc = damage_encoder.transform([damage_type])[0]
            cat_enc = category_encoder.transform([category])[0]
            severity_sq = severity ** 2
            part_sev = part_enc * severity
            damage_sev = damage_enc * severity
            features = [[part_enc, damage_enc, severity, cat_enc, severity_sq, part_sev, damage_sev]]
            cost = best_model.predict(features)[0]
            print(f"{part} - {damage_type} (Severity {severity}, {category}): ${cost:,.0f}")
        except ValueError as e:
            print(f"{part}: Not in training data - {e}")

    return best_model


def predict(part_name: str, damage_type: str, severity: int, car_category: str = DEFAULT_CATEGORY) -> float:
    """
    Predict repair cost for given inputs.

    Args:
        part_name: Name of the damaged car part.
        damage_type: Type of damage (e.g., bumper_dent).
        severity: Damage severity level (1-5).
        car_category: Vehicle category (e.g., sedan, luxury).
    """
    model = joblib.load(MODEL_DIR / "cost_estimator.pkl")
    part_encoder = joblib.load(MODEL_DIR / "part_encoder.pkl")
    damage_encoder = joblib.load(MODEL_DIR / "damage_encoder.pkl")
    category_encoder = joblib.load(MODEL_DIR / "category_encoder.pkl")

    part_enc = part_encoder.transform([part_name])[0]
    damage_enc = damage_encoder.transform([damage_type])[0]
    cat_enc = category_encoder.transform([car_category])[0]
    severity_sq = severity ** 2
    part_sev = part_enc * severity
    damage_sev = damage_enc * severity

    features = [[part_enc, damage_enc, severity, cat_enc, severity_sq, part_sev, damage_sev]]
    cost = model.predict(features)[0]

    print(f"Part: {part_name}")
    print(f"Damage Type: {damage_type}")
    print(f"Severity: {severity}/{MAX_SEVERITY}")
    print(f"Car Category: {car_category}")
    print(f"Estimated Cost: ${cost:,.0f}")

    return float(cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost Estimation Training v3")
    parser.add_argument("--data-path", type=Path, default=DATA_DIR / "repair_costs_v2.csv",
                        help="Path to training data CSV")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Test split ratio")

    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    train(config)
