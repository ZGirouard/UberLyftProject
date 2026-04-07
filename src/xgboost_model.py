import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class UberLyftXGBoost:

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def preprocess_data(self, df, fit=True):
        df = df.copy()
        df = df.dropna(subset=['price'])

        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'price']

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
                )
                if 'unknown' not in self.label_encoders[col].classes_:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'unknown'
                    )
                df[col] = self.label_encoders[col].transform(df[col])

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df.drop('timestamp', axis=1, inplace=True)
        
        return df

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = X_train.columns.tolist()

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
        }

        self.model = xgb.XGBRegressor(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
        }

        return metrics, y_pred

    def plot_feature_importance(self, top_n=20):
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        k = len(indices)

        plt.figure(figsize=(10, 8))
        plt.title('Top Feature Importances', fontsize=14, fontweight='bold')
        plt.barh(range(k), importance[indices])
        plt.yticks(range(k), [self.feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def calculate_confidence_intervals(model, X, y, n_iterations=100):
    metrics_list = {'RMSE': [], 'MAE': [], 'R2': []}

    for i in range(n_iterations):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        model.train(X_boot, y_boot)

        y_pred = model.predict(X_boot)
        metrics_list['RMSE'].append(np.sqrt(mean_squared_error(y_boot, y_pred)))
        metrics_list['MAE'].append(mean_absolute_error(y_boot, y_pred))
        metrics_list['R2'].append(r2_score(y_boot, y_pred))

    results = {}
    for metric in metrics_list:
        values = np.array(metrics_list[metric])
        results[metric] = {
            'mean': np.mean(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5)
        }
    
    return results

if __name__ == "__main__":
    print("-" * 50)
    print("XGBoost Model Training and Evaluation")
    print("-" * 50)

    data_path = "data/cab_rides.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    xgb_model = UberLyftXGBoost()

    print("Preprocessing data...")
    df_processed = xgb_model.preprocess_data(df, fit=True)

    X = df_processed.drop('price', axis=1)
    y = df_processed['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("-" * 50)
    print("Training XGBoost model...")
    xgb_model.train(X_train, y_train, X_test, y_test)

    print("-" * 50)
    print("\nEvaluating on test set...")
    metrics, y_pred = xgb_model.evaluate(X_test, y_test)

    print("-" * 50)
    print("Test Set Performance:")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAE: ${metrics['MAE']:.2f}")
    print(f"R²: {metrics['R2']:.4f}")

    print("-" * 50)
    print("Calculating confidence intervals...")
    ci_results = calculate_confidence_intervals(xgb_model, X_test, y_test)
    
    print("-" * 50)
    print("Confidence Intervals (95%):")
    for metric, values in ci_results.items():
        print(f"{metric}: {values['mean']:.2f} [{values['ci_lower']:.2f}, {values['ci_upper']:.2f}]")
    
    print("-" * 50)
    print("Plotting feature importance...")
    xgb_model.plot_feature_importance()
