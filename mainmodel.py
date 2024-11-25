import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import json
from typing import List, Dict
import xgboost as xgb
from xgboost.callback import EarlyStopping
import glob
import os

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
        
        # Check if innings exists and is not empty
        innings_data = match_data.get('innings', [])
        if not innings_data:
            print(f"No innings data found")
            return []
            
        for innings in innings_data:
            try:
                current_score = 0
                wickets = 0
                ball_data = []
            
                overs_data = innings.get('overs', [])
                if not overs_data:
                    print(f"No overs data found in innings")
                    continue
                
                for over in overs_data:
                    try:
                        over_num = over.get('over', 0)
                        deliveries = over.get('deliveries', [])
                        
                        if not deliveries:
                            print(f"No deliveries in over {over_num}")
                            continue

                        for delivery in deliveries:
                            try:
                                runs_data = delivery.get('runs', {})
                                runs = runs_data.get('total', 0)
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
                            except Exception as e:
                                print(f"Error processing delivery: {str(e)}")
                                continue
                                
                    except Exception as e:
                        print(f"Error processing over: {str(e)}")
                        continue
                
                if ball_data:  # Only process if we have valid ball data
                    final_score = current_score
                    for i in range(len(ball_data)):
                        if i < 30:  # Skip very early stage predictions
                            continue
                            
                        current_state = ball_data[:i+1]
                        current_over = current_state[-1]['over']
                        
                        try:
                            features = self._calculate_features(
                                current_state,
                                innings.get('powerplays', []),
                                match_data.get('info', {}).get('overs', 50)  # Default to 50 overs if not specified
                            )
                            
                            features['final_score'] = final_score
                            processed_data.append(features)
                        except Exception as e:
                            print(f"Error calculating features: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Error processing innings: {str(e)}")
                continue
                
        return processed_data
    
    def _calculate_features(self, current_state: List[Dict], powerplays: List[Dict], total_overs: int) -> Dict:
        """Calculate features from current match state with error handling"""
        try:
            current_ball = current_state[-1]
            
            # Get last 5 overs data if available
            last_30_balls = current_state[-30:] if len(current_state) >= 30 else current_state
            
            # Calculate run rate and recent performance
            runs_last_5 = sum(ball.get('runs', 0) for ball in last_30_balls)
            wickets_last_5 = sum(ball.get('wicket', 0) for ball in last_30_balls)
            
            # Determine if in powerplay
            current_over = current_ball.get('over', 0)
            in_powerplay = any(
                pp.get('from', 0) <= current_over <= pp.get('to', 0)
                for pp in powerplays
            ) if powerplays else False
            
            # Calculate balls remaining
            total_balls = total_overs * 6
            balls_played = len(current_state)
            balls_remaining = total_balls - balls_played
            
            # Calculate run rate safely
            run_rate = (current_ball['cumulative_score'] / max(balls_played, 1)) * 6
            
            return {
                'current_score': current_ball.get('cumulative_score', 0),
                'wickets_fallen': current_ball.get('wickets', 0),
                'current_over': current_over,
                'runs_last_5_overs': runs_last_5,
                'wickets_last_5_overs': wickets_last_5,
                'powerplay_active': 1 if in_powerplay else 0,
                'run_rate': run_rate,
                'balls_remaining': balls_remaining,
                'batting_team_avg_score': 250  # Default value
            }
        except Exception as e:
            print(f"Error in _calculate_features: {str(e)}")
            raise
    
    def prepare_training_data(self, json_files: List[str]):
        """
        Prepare training data from multiple JSON matches
        """
        all_samples = []
        processed_files = 0
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    match_data = json.load(f)
                samples = self.process_json_match(match_data)
                if samples:
                    all_samples.extend(samples)
                    processed_files += 1
                    print(f"Successfully processed {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"\nSuccessfully processed {processed_files} out of {len(json_files)} files")
        
        if not all_samples:
            raise ValueError("No valid training samples were generated")
        df = pd.DataFrame(all_samples)
        return df[self.features], df['final_score']
    
    def train(self, X, y):
        """Train the prediction model"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
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
        try:
            # Process current match state
            processed_states = self.process_json_match(current_match_json)
            if not processed_states:
                raise ValueError("No valid states could be processed from the match data")
            
            current_state = processed_states[-1]  # Get latest state
            
            # Prepare features
            X = pd.DataFrame([current_state])[self.features]
            
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            
            # Make prediction
            predicted_score = self.model.predict(X)[0]
            
            # Calculate confidence interval (using model's feature importances)
            feature_importances = self.model.feature_importances_
            
            # Handle case where there's only one sample
            if X.shape[0] == 1:
                # Use historical standard deviations or a reasonable default
                std_dev = np.array([10.0] * len(self.features))  # Default value
            else:
                std_dev = np.std(X, axis=0)
            
            prediction_uncertainty = np.sum(feature_importances * std_dev) * 2
            
            # Calculate run rates safely
            balls_remaining = max(current_state['balls_remaining'], 1)  # Avoid division by zero
            required_run_rate = (
                (predicted_score - current_state['current_score']) 
                / (balls_remaining / 6)
            )
            
            return {
                'predicted_score': round(predicted_score),
                'confidence_interval': (
                    round(predicted_score - prediction_uncertainty),
                    round(predicted_score + prediction_uncertainty)
                ),
                'current_run_rate': current_state['run_rate'],
                'required_run_rate': max(0, required_run_rate)  # Ensure non-negative
            }
            
        except Exception as e:
            print(f"Error in predict_live_match: {str(e)}")
            raise

def main():
    predictor = CricketJSONPredictor()

    # Load test match data
    with open('test/1086066.json', 'r') as f:
        match_data = json.load(f)
    
    # Process and train on data
    # Get all JSON files from the dataset directory
    dataset_path = 'dataset/*.json'
    json_files = glob.glob(dataset_path)
    
    if not json_files:
        print(f"No JSON files found in {dataset_path}")
        return
        
    print(f"Found {len(json_files)} match files")
    
    try:
        # Train the model with all available data 
        training_files = json_files[:]
        
        print("\nTraining model...")
        X, y = predictor.prepare_training_data(training_files)
        print(f"Training with {len(X)} samples from {len(training_files)} matches")
        
        metrics = predictor.train(X, y)
        
        print("\nModel Performance Metrics:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R²: {metrics['r2']:.2f}")
        
        # Make prediction for the last match
        print("\nMaking prediction for the given match...")
        prediction = predictor.predict_live_match(match_data)
        
        print("\nPrediction:")
        print(f"Predicted Final Score: {prediction['predicted_score']}")
        print(f"Confidence Interval: {prediction['confidence_interval']}")
        print(f"Current Run Rate: {prediction['current_run_rate']:.2f}")
        print(f"Required Run Rate: {prediction['required_run_rate']:.2f}")
        
        # Save the trained model
        import pickle
        with open('pickle/cricket_predictor.pkl', 'wb') as f:
            pickle.dump(predictor, f)
        print("\nModel saved as 'cricket_predictor.pkl'")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
