import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "severity-model" / "dataset" / "detailed_repair_costs.csv"
MODEL_DIR = ROOT / "severity-model" / "models"
MODEL_DIR.mkdir(exist_ok=True)
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Total samples: {len(df)}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData statistics:\n{df.describe()}")
part_encoder = LabelEncoder()
segment_encoder = LabelEncoder()
df['Part_Encoded'] = part_encoder.fit_transform(df['Part_Name'])
df['Segment_Encoded'] = segment_encoder.fit_transform(df['Car_Segment'])
X = df[['Part_Encoded', 'Severity', 'Segment_Encoded']]
y = df['Estimated_Cost']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"\nDataset split:")
print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}")
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train, y_train)
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=1
)
gb_model.fit(X_train, y_train)
def evaluate_model(model, X, y, set_name):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"\n{set_name} Set:")
    print(f"  MAE:  ₪{mae:.2f}")
    print(f"  RMSE: ₪{rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}
print("\n" + "="*50)
print("Random Forest Results:")
print("="*50)
rf_train_metrics = evaluate_model(rf_model, X_train, y_train, "Train")
rf_val_metrics = evaluate_model(rf_model, X_val, y_val, "Validation")
rf_test_metrics = evaluate_model(rf_model, X_test, y_test, "Test")
print("\n" + "="*50)
print("Gradient Boosting Results:")
print("="*50)
gb_train_metrics = evaluate_model(gb_model, X_train, y_train, "Train")
gb_val_metrics = evaluate_model(gb_model, X_val, y_val, "Validation")
gb_test_metrics = evaluate_model(gb_model, X_test, y_test, "Test")
if rf_val_metrics['mae'] < gb_val_metrics['mae']:
    best_model = rf_model
    best_name = "Random Forest"
    best_metrics = rf_test_metrics
else:
    best_model = gb_model
    best_name = "Gradient Boosting"
    best_metrics = gb_test_metrics
print(f"\n Best model: {best_name}")
print(f"   Test MAE: ₪{best_metrics['mae']:.2f}")
print(f"   Test R²:  {best_metrics['r2']:.4f}")
joblib.dump(best_model, MODEL_DIR / 'cost_estimator.pkl')
joblib.dump(part_encoder, MODEL_DIR / 'part_encoder.pkl')
joblib.dump(segment_encoder, MODEL_DIR / 'segment_encoder.pkl')
other_model = gb_model if best_name == "Random Forest" else rf_model
other_name = "gb" if best_name == "Random Forest" else "rf"
joblib.dump(other_model, MODEL_DIR / f'cost_estimator_{other_name}.pkl')
metadata = {
    'best_model': best_name,
    'test_metrics': best_metrics,
    'parts': part_encoder.classes_.tolist(),
    'segments': segment_encoder.classes_.tolist(),
    'severity_range': [1, 5],
    'feature_names': ['Part_Encoded', 'Severity', 'Segment_Encoded']
}
with open(MODEL_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\n Models and encoders saved to: {MODEL_DIR}")
print("\n" + "="*50)
print("Example Predictions:")
print("="*50)
examples = [
    ("Front Bumper", 3, "Family"),
    ("Headlight", 5, "Luxury"),
    ("Door Handle", 1, "Micro"),
    ("Hood", 4, "SUV")
]
for part, severity, segment in examples:
    part_enc = part_encoder.transform([part])[0]
    seg_enc = segment_encoder.transform([segment])[0]
    cost = best_model.predict([[part_enc, severity, seg_enc]])[0]
    print(f"{part} (Severity {severity}, {segment}): ₪{cost:.0f}")
print("\n Training complete!")