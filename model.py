"""
Engagement Prediction Model Module
Builds and manages machine learning models for predicting social media engagement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path

from data_processing import extract_time_features, prepare_features_for_model

MODEL_DIR = Path(__file__).parent / "models"


class EngagementPredictor:
    """Predicts engagement metrics (likes, comments) based on post characteristics."""

    def __init__(self):
        self.likes_model = None
        self.comments_model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['hour', 'day_of_week', 'is_weekend', 'is_video']
        self.is_trained = False
        self.model_metrics = {}

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for training/prediction."""
        df = df.copy()

        # Extract time features
        df = extract_time_features(df)

        # Convert is_video to numeric
        if df['is_video'].dtype == 'object':
            df['is_video'] = df['is_video'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
        else:
            df['is_video'] = df['is_video'].astype(int)

        X = df[self.feature_columns].copy()
        y_likes = df['likes'].copy() if 'likes' in df.columns else None
        y_comments = df['comments'].copy() if 'comments' in df.columns else None

        return X, y_likes, y_comments

    def train(self, df: pd.DataFrame, model_type: str = 'random_forest') -> dict:
        """Train the engagement prediction models."""
        X, y_likes, y_comments = self.prepare_data(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Select model type
        if model_type == 'random_forest':
            self.likes_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            self.comments_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        elif model_type == 'gradient_boosting':
            self.likes_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
            self.comments_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
        elif model_type == 'linear':
            self.likes_model = Ridge(alpha=1.0)
            self.comments_model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train models
        self.likes_model.fit(X_scaled, y_likes)
        self.comments_model.fit(X_scaled, y_comments)

        # Calculate metrics using cross-validation
        likes_cv_scores = cross_val_score(self.likes_model, X_scaled, y_likes, cv=min(5, len(df)), scoring='r2')
        comments_cv_scores = cross_val_score(self.comments_model, X_scaled, y_comments, cv=min(5, len(df)), scoring='r2')

        # Calculate predictions for full dataset
        likes_pred = self.likes_model.predict(X_scaled)
        comments_pred = self.comments_model.predict(X_scaled)

        self.model_metrics = {
            'likes': {
                'mae': round(mean_absolute_error(y_likes, likes_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_likes, likes_pred)), 2),
                'r2': round(r2_score(y_likes, likes_pred), 4),
                'cv_r2_mean': round(likes_cv_scores.mean(), 4),
                'cv_r2_std': round(likes_cv_scores.std(), 4)
            },
            'comments': {
                'mae': round(mean_absolute_error(y_comments, comments_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_comments, comments_pred)), 2),
                'r2': round(r2_score(y_comments, comments_pred), 4),
                'cv_r2_mean': round(comments_cv_scores.mean(), 4),
                'cv_r2_std': round(comments_cv_scores.std(), 4)
            },
            'model_type': model_type,
            'n_samples': len(df)
        }

        self.is_trained = True
        return self.model_metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict engagement for new posts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X, _, _ = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)

        predictions = df.copy()
        predictions['predicted_likes'] = np.maximum(0, self.likes_model.predict(X_scaled)).astype(int)
        predictions['predicted_comments'] = np.maximum(0, self.comments_model.predict(X_scaled)).astype(int)
        predictions['predicted_engagement'] = predictions['predicted_likes'] + predictions['predicted_comments']

        return predictions

    def predict_single(self, hour: int, day_of_week: int, is_weekend: int, is_video: int) -> dict:
        """Predict engagement for a single hypothetical post."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        features = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'is_video': [is_video]
        })

        features_scaled = self.scaler.transform(features)

        predicted_likes = max(0, int(self.likes_model.predict(features_scaled)[0]))
        predicted_comments = max(0, int(self.comments_model.predict(features_scaled)[0]))

        return {
            'predicted_likes': predicted_likes,
            'predicted_comments': predicted_comments,
            'predicted_engagement': predicted_likes + predicted_comments
        }

    def get_feature_importance(self) -> dict:
        """Get feature importance from the trained models."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = {}

        if hasattr(self.likes_model, 'feature_importances_'):
            importance['likes'] = dict(zip(
                self.feature_columns,
                [round(x, 4) for x in self.likes_model.feature_importances_]
            ))
            importance['comments'] = dict(zip(
                self.feature_columns,
                [round(x, 4) for x in self.comments_model.feature_importances_]
            ))
        else:
            # For linear models, use coefficients
            importance['likes'] = dict(zip(
                self.feature_columns,
                [round(x, 4) for x in self.likes_model.coef_]
            ))
            importance['comments'] = dict(zip(
                self.feature_columns,
                [round(x, 4) for x in self.comments_model.coef_]
            ))

        return importance

    def get_optimal_posting_time(self) -> dict:
        """Find the optimal posting time based on predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        best_engagement = 0
        best_config = {}

        # Test all combinations
        for hour in range(24):
            for day in range(7):
                is_weekend = 1 if day >= 5 else 0
                for is_video in [0, 1]:
                    pred = self.predict_single(hour, day, is_weekend, is_video)
                    if pred['predicted_engagement'] > best_engagement:
                        best_engagement = pred['predicted_engagement']
                        best_config = {
                            'hour': hour,
                            'day_of_week': day,
                            'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day],
                            'is_weekend': bool(is_weekend),
                            'is_video': bool(is_video),
                            'predicted_engagement': pred['predicted_engagement'],
                            'predicted_likes': pred['predicted_likes'],
                            'predicted_comments': pred['predicted_comments']
                        }

        return best_config

    def save_model(self, filename: str = "engagement_model.pkl"):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / filename

        model_data = {
            'likes_model': self.likes_model,
            'comments_model': self.comments_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        return str(model_path)

    def load_model(self, filename: str = "engagement_model.pkl"):
        """Load a trained model from disk."""
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.likes_model = model_data['likes_model']
        self.comments_model = model_data['comments_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data['model_metrics']
        self.is_trained = True


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compare different model types on the dataset."""
    model_types = ['random_forest', 'gradient_boosting', 'linear']
    results = []

    for model_type in model_types:
        predictor = EngagementPredictor()
        metrics = predictor.train(df, model_type=model_type)

        results.append({
            'Model': model_type,
            'Likes MAE': metrics['likes']['mae'],
            'Likes R²': metrics['likes']['r2'],
            'Comments MAE': metrics['comments']['mae'],
            'Comments R²': metrics['comments']['r2']
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the model with sample data
    import database

    database.init_database()

    csv_path = Path(__file__).parent / "insta_dummy_data.csv"
    if csv_path.exists():
        database.load_csv_to_database(str(csv_path), replace_existing=True)

    df = database.get_all_posts()

    if len(df) > 0:
        predictor = EngagementPredictor()
        metrics = predictor.train(df)

        print("Model Training Results:")
        print(f"  Likes R²: {metrics['likes']['r2']}")
        print(f"  Comments R²: {metrics['comments']['r2']}")

        print("\nOptimal Posting Time:")
        optimal = predictor.get_optimal_posting_time()
        print(f"  Day: {optimal['day_name']}")
        print(f"  Hour: {optimal['hour']}:00")
        print(f"  Content Type: {'Video' if optimal['is_video'] else 'Image'}")
        print(f"  Predicted Engagement: {optimal['predicted_engagement']}")
