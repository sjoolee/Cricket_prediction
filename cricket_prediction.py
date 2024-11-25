import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import json
from typing import List, Dict
import xgboost as xgb
from xgboost.callback import EarlyStopping
import glob

class CricketJSONPredictor:
    def __init__(self):
        self.model = None
        self.features = [
            'current_score', 
            'wickets_fallen',
            'current_over',
            'runs_last_5_overs',
            'wickets_last_5_overs',
            'powerplay_active',
            'run_rate',
            'balls_remaining',
            'batting_team_avg_score'
        ]
    
    def process_json_match(self, match_data: Dict) -> List[Dict]:
        """
        Process JSON data into training samples
        """
        processed_data = []
        
        for innings in match_data.get('innings', []):
            current_score = 0
            wickets = 0
            ball_data = []
            
            for over in innings.get('overs', []):
                over_num = over['over']
                
                for delivery in over.get('deliveries', []):
                    runs = delivery['runs']['total']
                    current_score += runs
                    
                    # Check for wicket
                    if 'wickets' in delivery:
                        wickets += len(delivery['wickets'])
                    
                    ball_data.append({
                        'over': over_num,
                        'runs': runs,
                        'wicket': 1 if 'wickets' in delivery else 0,
                        'cumulative_score': current_score,
                        'wickets': wickets
                    })
            
            final_score = current_score
            for i in range(len(ball_data)):
                if i < 30:  # Skip very early stage predictions
                    continue
                    
                current_state = ball_data[:i+1]
                current_over = current_state[-1]['over']
                
                # Calculate features
                features = self._calculate_features(
                    current_state,
                    innings.get('powerplays', []),
                    match_data['info']['overs']
                )
                
                features['final_score'] = final_score
                processed_data.append(features)
        
        return processed_data
    
    def _calculate_features(self, current_state: List[Dict], powerplays: List[Dict], total_overs: int) -> Dict:
        """Calculate features from current match state"""
        current_ball = current_state[-1]
        
        # Get last 5 overs data if available
        last_30_balls = current_state[-30:] if len(current_state) >= 30 else current_state
        
        # Calculate run rate and recent performance
        runs_last_5 = sum(ball['runs'] for ball in last_30_balls)
        wickets_last_5 = sum(ball['wicket'] for ball in last_30_balls)
        
        # Determine if in powerplay
        current_over = current_ball['over']
        in_powerplay = any(
            pp['from'] <= current_over <= pp['to'] 
            for pp in powerplays
        )
        
        # Calculate balls remaining
        total_balls = total_overs * 6
        balls_played = len(current_state)
        balls_remaining = total_balls - balls_played
        
        return {
            'current_score': current_ball['cumulative_score'],
            'wickets_fallen': current_ball['wickets'],
            'current_over': current_over,
            'runs_last_5_overs': runs_last_5,
            'wickets_last_5_overs': wickets_last_5,
            'powerplay_active': 1 if in_powerplay else 0,
            'run_rate': (current_ball['cumulative_score'] / balls_played) * 6,
            'balls_remaining': balls_remaining,
            'batting_team_avg_score': 250  # Can be updated with historical data
        }
    
    def prepare_training_data(self, json_files: List[str]):
        """
        Prepare training data from multiple JSON match files
        """
        all_samples = []
        
        for file_path in json_files:
            with open(file_path, 'r') as f:
                match_data = json.load(f)
                samples = self.process_json_match(match_data)
                all_samples.extend(samples)
        df = pd.DataFrame(all_samples)
        return df[self.features], df['final_score']
    
    def train(self, X, y):
        """
        Train the prediction model
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            early_stopping_rounds = 10
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],  # Explicitly pass the callback
            verbose=False
        )
        
        val_predictions = self.model.predict(X_val)
        metrics = {
            'rmse': np.sqrt(((val_predictions - y_val) ** 2).mean()),
            'mae': np.abs(val_predictions - y_val).mean(),
            'r2': 1 - ((y_val - val_predictions) ** 2).sum() / ((y_val - y_val.mean()) ** 2).sum()
        }
        return metrics

    
    def predict_live_match(self, current_match_json: Dict) -> Dict:
        """
        Make prediction for an ongoing match
        """
        # current match state
        current_state = self.process_json_match(current_match_json)[-1]  # Get latest state
        
        # Prepare features
        X = pd.DataFrame([current_state])[self.features]
        
        # Make prediction
        predicted_score = self.model.predict(X)[0]
        
        # Calculate confidence interval (using model's feature importances)
        feature_importances = self.model.feature_importances_
        prediction_uncertainty = np.sum(feature_importances * np.std(X, axis=0)) * 2
        
        return {
            'predicted_score': round(predicted_score),
            'confidence_interval': (
                round(predicted_score - prediction_uncertainty),
                round(predicted_score + prediction_uncertainty)
            ),
            'current_run_rate': current_state['run_rate'],
            'required_run_rate': (
                predicted_score - current_state['current_score']
            ) / (current_state['balls_remaining'] / 6)
        }

def main():
    predictor = CricketJSONPredictor()
    
    dataset_path = 'dataset/*.json'
    json_files = glob.glob(dataset_path)
    
    if not json_files:
        print(f"No JSON files found in {dataset_path}")
        return
        
    print(f"Found {len(json_files)} match files")
    
    # Split files into training and validation sets
    train_files, val_files = train_test_split(
        json_files, 
        test_size=0.2, 
        random_state=42
    )
    
    # Load and process all training data
    all_X = []
    all_y = []
    
    print("Processing training files...")
    for file in train_files:
        try:
            X, y = predictor.prepare_training_data([file])
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Combine all training data
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    print(f"Training with {len(X_combined)} samples")
    
    # Train the model
    metrics = predictor.train(X_combined, y_combined)
    
    print("\nModel Performance Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")
    
    # Validate on dataset
    print("\nValidating on dataset...")
    validation_predictions = []
    actual_scores = []
    
    for file in val_files:
        try:
            with open(file, 'r') as f:
                match_data = json.load(f)
            
            # Get actual final score
            final_score = match_data['innings'][0]['overs'][-1]['deliveries'][-1]['runs']['total']
            
            # Make prediction
            prediction = predictor.predict_live_match(match_data)
            
            validation_predictions.append(prediction['predicted_score'])
            actual_scores.append(final_score)
            
        except Exception as e:
            print(f"Error validating {file}: {str(e)}")
            continue

    val_predictions = np.array(validation_predictions)
    val_actuals = np.array(actual_scores)
    
    val_metrics = {
        'rmse': np.sqrt(((val_predictions - val_actuals) ** 2).mean()),
        'mae': np.abs(val_predictions - val_actuals).mean(),
        'r2': 1 - ((val_actuals - val_predictions) ** 2).sum() / ((val_actuals - val_actuals.mean()) ** 2).sum()
    }
    
    print("\nValidation Set Metrics:")
    print(f"RMSE: {val_metrics['rmse']:.2f}")
    print(f"MAE: {val_metrics['mae']:.2f}")
    print(f"R²: {val_metrics['r2']:.2f}")
    
    
    import pickle
    with open('pickle/cricket_predictor_model1.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    print("\nModel saved as 'cricket_predictor_model1.pkl'")

if __name__ == "__main__":
    main()
