"""
XGBoost Machine Learning Strategy

Uses gradient boosting to predict price movements and generate trading signals
Incorporates feature engineering and model retraining
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    # Create dummy classes for type hints when imports fail
    class StandardScaler:
        pass
    class RobustScaler:
        pass

from .base_strategy import BaseStrategy

class XGBoostStrategy(BaseStrategy):
    """
    XGBoost-based trading strategy featuring:
    - Comprehensive feature engineering
    - Market regime detection
    - Model retraining and validation
    - Risk-aware position sizing
    - Multiple prediction horizons
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'prediction_horizon': 5,      # Days ahead to predict
            'feature_lookback': 60,       # Days of features to use
            'min_training_samples': 1000, # Minimum samples for training
            'retrain_frequency': 30,      # Retrain every N days
            'confidence_threshold': 0.6,  # Minimum prediction confidence
            'position_size_pct': 0.06,    # 6% of portfolio per position
            'max_positions': 8,           # Maximum concurrent positions
            'stop_loss_pct': 0.03,        # 3% stop loss
            'take_profit_pct': 0.06,      # 6% take profit
            'model_validation_split': 0.2, # 20% for validation
            'feature_selection_top_k': 50, # Top K features to use
            'xgb_params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
        
        if not XGBOOST_AVAILABLE:
            self.logger.error("XGBoost not available. Install with: pip install xgboost scikit-learn")
            
        self.models = {}  # Store models per symbol
        self.scalers = {}  # Store scalers per symbol
        self.feature_names = {}  # Store feature names per symbol
        self.last_training = {}  # Track last training date per symbol
        self.model_performance = {}  # Track model performance metrics
        
        # Model storage directory
        self.model_dir = Path("models/xgboost")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate ML-based trading signals"""
        
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available, skipping ML strategy")
            return []
        
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_prices or data.empty:
                continue
            
            try:
                signal = self._analyze_ml_opportunity(
                    symbol, data, current_prices[symbol], 
                    current_data.get(symbol, {}), current_date, portfolio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} with XGBoost: {e}")
                continue
        
        return signals
    
    def _analyze_ml_opportunity(self, symbol: str, data: pd.DataFrame,
                               current_price: float, current_bar: Dict[str, Any],
                               current_date: datetime, portfolio: Any = None) -> Optional[Dict[str, Any]]:
        """Analyze ML trading opportunity for a single symbol"""
        
        min_data_length = self.parameters['feature_lookback'] + self.parameters['prediction_horizon'] + 100
        if len(data) < min_data_length:
            return None
        
        # Check if model needs training/retraining
        needs_training = self._check_training_needed(symbol, current_date)
        
        if needs_training:
            success = self._train_model(symbol, data, current_date)
            if not success:
                return None
        
        # Check if model exists
        if symbol not in self.models:
            return None
        
        # Generate features for current state
        features = self._engineer_features(data, for_prediction=True)
        if features is None or len(features) == 0:
            return None
        
        # Make prediction
        prediction_result = self._make_prediction(symbol, features)
        if prediction_result is None:
            return None
        
        prediction, confidence = prediction_result
        
        # Check confidence threshold
        if confidence < self.parameters['confidence_threshold']:
            return None
        
        # Determine action based on prediction
        if prediction > 0.6:  # Bullish prediction
            action = 'buy'
        elif prediction < 0.4:  # Bearish prediction
            action = 'sell'
        else:
            return None  # Neutral prediction
        
        # Portfolio constraints
        if not self._check_portfolio_constraints(symbol, action, portfolio):
            return None
        
        # Calculate position size
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price}) if portfolio else 100000
        position_size = self.calculate_position_size(symbol, current_price, portfolio_value)
        
        # Risk management
        stop_loss, take_profit = self._calculate_ml_stops(current_price, action, confidence)
        
        return {
            'symbol': symbol,
            'action': action,
            'quantity': position_size,
            'price': current_price,
            'strategy': 'xgboost_ml',
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'prediction': prediction,
            'model_performance': self.model_performance.get(symbol, {}),
            'holding_period_target': timedelta(days=self.parameters['prediction_horizon']),
            'ml_metadata': {
                'model_age_days': (current_date - self.last_training.get(symbol, current_date)).days,
                'feature_count': len(features),
                'prediction_horizon': self.parameters['prediction_horizon']
            }
        }
    
    def _check_training_needed(self, symbol: str, current_date: datetime) -> bool:
        """Check if model needs (re)training"""
        
        # No model exists
        if symbol not in self.models:
            return True
        
        # Check training frequency
        last_train = self.last_training.get(symbol)
        if last_train is None:
            return True
        
        days_since_training = (current_date - last_train).days
        return days_since_training >= self.parameters['retrain_frequency']
    
    def _train_model(self, symbol: str, data: pd.DataFrame, current_date: datetime) -> bool:
        """Train XGBoost model for the symbol"""
        
        try:
            self.logger.info(f"Training XGBoost model for {symbol}")
            
            # Prepare training data
            features, targets = self._prepare_training_data(data)
            
            if len(features) < self.parameters['min_training_samples']:
                self.logger.warning(f"Insufficient training data for {symbol}: {len(features)} samples")
                return False
            
            # Feature selection and scaling
            features_scaled, feature_names = self._preprocess_features(symbol, features)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets, 
                test_size=self.parameters['model_validation_split'],
                random_state=42,
                stratify=targets if len(np.unique(targets)) > 1 else None
            )
            
            # Train XGBoost model
            model = xgb.XGBClassifier(**self.parameters['xgb_params'])
            
            # Fit with early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Validate model
            val_predictions = model.predict(X_val)
            val_probabilities = model.predict_proba(X_val)
            
            performance = {
                'accuracy': accuracy_score(y_val, val_predictions),
                'precision': precision_score(y_val, val_predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_val, val_predictions, average='weighted', zero_division=0),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'training_date': current_date,
                'feature_count': X_train.shape[1]
            }
            
            # Store model and metadata
            self.models[symbol] = model
            self.feature_names[symbol] = feature_names
            self.last_training[symbol] = current_date
            self.model_performance[symbol] = performance
            
            # Save model to disk
            self._save_model(symbol, model)
            
            self.logger.info(f"XGBoost model trained for {symbol} - Accuracy: {performance['accuracy']:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model for {symbol}: {e}")
            return False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare features and targets for training"""
        
        # Generate features for all historical data
        features_df = self._engineer_features(data, for_prediction=False)
        
        # Create targets (future returns)
        horizon = self.parameters['prediction_horizon']
        future_returns = data['close'].pct_change(horizon).shift(-horizon)
        
        # Convert to classification targets
        # 0: Down, 1: Neutral, 2: Up
        targets = np.where(future_returns > 0.02, 2,  # Up > 2%
                          np.where(future_returns < -0.02, 0, 1))  # Down < -2%, else Neutral
        
        # Align features and targets
        valid_indices = ~(np.isnan(targets) | features_df.isnull().any(axis=1))
        
        features_clean = features_df[valid_indices]
        targets_clean = targets[valid_indices]
        
        return features_clean, targets_clean
    
    def _engineer_features(self, data: pd.DataFrame, for_prediction: bool = False) -> Optional[pd.DataFrame]:
        """Engineer comprehensive features for ML model"""
        
        if len(data) < self.parameters['feature_lookback']:
            return None
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns_1d'] = data['close'].pct_change(1)
        features['returns_3d'] = data['close'].pct_change(3)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_10d'] = data['close'].pct_change(10)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # Volatility features
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_10d'] = features['returns_1d'].rolling(10).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['rsi_7'] = self._calculate_rsi(data['close'], 7)
        
        # Moving averages and ratios
        features['sma_5'] = data['close'].rolling(5).mean()
        features['sma_10'] = data['close'].rolling(10).mean()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        
        features['price_to_sma_5'] = data['close'] / features['sma_5']
        features['price_to_sma_20'] = data['close'] / features['sma_20']
        features['price_to_sma_50'] = data['close'] / features['sma_50']
        
        features['sma_5_to_20'] = features['sma_5'] / features['sma_20']
        features['sma_20_to_50'] = features['sma_20'] / features['sma_50']
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'], 20, 2)
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(data['close'])
        features['macd_line'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram
        
        # Volume features
        features['volume_ratio_5d'] = data['volume'] / data['volume'].rolling(5).mean()
        features['volume_ratio_20d'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_sma_5'] = data['volume'].rolling(5).mean()
        features['volume_sma_20'] = data['volume'].rolling(20).mean()
        
        # Price-volume features
        features['price_volume'] = data['close'] * data['volume']
        features['vwap_5'] = (features['price_volume'].rolling(5).sum() / 
                             data['volume'].rolling(5).sum())
        features['vwap_20'] = (features['price_volume'].rolling(20).sum() / 
                              data['volume'].rolling(20).sum())
        
        # High-low features
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Gap features
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        features['gap_abs'] = np.abs(features['gap'])
        
        # Trend features
        features['trend_5d'] = np.where(features['sma_5'] > features['sma_5'].shift(5), 1, 0)
        features['trend_20d'] = np.where(features['sma_20'] > features['sma_20'].shift(20), 1, 0)
        
        # Market microstructure (approximate)
        features['bid_ask_spread'] = (data['high'] - data['low']) / data['close']  # Proxy
        features['order_flow'] = np.where(data['close'] > data['open'], 1, -1)  # Simplified
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_1d_lag_{lag}'] = features['returns_1d'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio_5d'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
        
        # Time-based features
        features['day_of_week'] = pd.to_datetime(data.index).dayofweek
        features['month'] = pd.to_datetime(data.index).month
        features['quarter'] = pd.to_datetime(data.index).quarter
        
        # Statistical features
        features['returns_skew_20d'] = features['returns_1d'].rolling(20).skew()
        features['returns_kurt_20d'] = features['returns_1d'].rolling(20).kurt()
        
        # Regime detection features
        features['volatility_regime'] = np.where(features['volatility_20d'] > 
                                               features['volatility_20d'].rolling(60).mean(), 1, 0)
        
        # Drop rows with too many NaN values
        features = features.dropna(thresh=len(features.columns) * 0.7)
        
        # For prediction, return only the last row
        if for_prediction:
            if len(features) > 0:
                return features.iloc[-1:].fillna(method='ffill').fillna(0)
            else:
                return None
        
        # For training, return all data
        return features.fillna(method='ffill').fillna(0)
    
    def _preprocess_features(self, symbol: str, features: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features with scaling and selection"""
        
        # Feature selection based on importance (if model exists)
        if symbol in self.models and hasattr(self.models[symbol], 'feature_importances_'):
            # Use feature importance from previous model
            importances = self.models[symbol].feature_importances_
            feature_names = self.feature_names.get(symbol, features.columns.tolist())
            
            if len(importances) == len(feature_names):
                # Select top K features
                top_k = self.parameters['feature_selection_top_k']
                top_indices = np.argsort(importances)[-top_k:]
                selected_features = [feature_names[i] for i in top_indices if i < len(features.columns)]
                features_selected = features[selected_features]
            else:
                features_selected = features
        else:
            features_selected = features
        
        # Scale features
        if symbol not in self.scalers:
            self.scalers[symbol] = RobustScaler()  # More robust to outliers than StandardScaler
            features_scaled = self.scalers[symbol].fit_transform(features_selected)
        else:
            features_scaled = self.scalers[symbol].transform(features_selected)
        
        feature_names = features_selected.columns.tolist()
        
        return features_scaled, feature_names
    
    def _make_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Make prediction using trained model"""
        
        if symbol not in self.models or symbol not in self.scalers:
            return None
        
        try:
            # Preprocess features
            features_scaled, _ = self._preprocess_features(symbol, features)
            
            # Make prediction
            model = self.models[symbol]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Convert to single probability (0 = bearish, 1 = bullish)
            if len(probabilities) == 3:  # [down, neutral, up]
                # Combine down and up, ignore neutral
                bearish_prob = probabilities[0]
                bullish_prob = probabilities[2]
                total_directional = bearish_prob + bullish_prob
                
                if total_directional > 0:
                    prediction = bullish_prob / total_directional
                    confidence = total_directional  # Strength of directional prediction
                else:
                    prediction = 0.5
                    confidence = 0.0
            else:
                prediction = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def _calculate_ml_stops(self, current_price: float, action: str, confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on ML confidence"""
        
        # Adjust stop/profit based on confidence
        base_stop = self.parameters['stop_loss_pct']
        base_profit = self.parameters['take_profit_pct']
        
        # Higher confidence = wider profit target, tighter stop
        confidence_factor = max(0.5, confidence)
        
        stop_pct = base_stop / confidence_factor
        profit_pct = base_profit * confidence_factor
        
        if action == 'buy':
            stop_loss = current_price * (1 - stop_pct)
            take_profit = current_price * (1 + profit_pct)
        else:  # sell
            stop_loss = current_price * (1 + stop_pct)
            take_profit = current_price * (1 - profit_pct)
        
        return stop_loss, take_profit
    
    def _check_portfolio_constraints(self, symbol: str, action: str, portfolio: Any) -> bool:
        """Check portfolio constraints for ML strategy"""
        if portfolio is None:
            return True
        
        # Check maximum positions
        current_positions = len([pos for pos in portfolio.positions.values() 
                               if pos['quantity'] != 0])
        
        if current_positions >= self.parameters['max_positions']:
            # Only allow closing positions
            if symbol in portfolio.positions:
                current_qty = portfolio.positions[symbol]['quantity']
                if action == 'sell' and current_qty > 0:
                    return True
                elif action == 'buy' and current_qty < 0:
                    return True
            return False
        
        return True
    
    def _save_model(self, symbol: str, model) -> None:
        """Save trained model to disk"""
        try:
            model_path = self.model_dir / f"{symbol}_xgboost_model.joblib"
            scaler_path = self.model_dir / f"{symbol}_scaler.joblib"
            
            joblib.dump(model, model_path)
            if symbol in self.scalers:
                joblib.dump(self.scalers[symbol], scaler_path)
                
        except Exception as e:
            self.logger.error(f"Error saving model for {symbol}: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load trained model from disk"""
        try:
            model_path = self.model_dir / f"{symbol}_xgboost_model.joblib"
            scaler_path = self.model_dir / f"{symbol}_scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                return True
            
        except Exception as e:
            self.logger.error(f"Error loading model for {symbol}: {e}")
        
        return False
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for ML strategy"""
        
        # Base position size
        target_value = portfolio_value * self.parameters['position_size_pct']
        position_size = int(target_value / price)
        
        # Adjust based on model confidence if available
        if symbol in self.model_performance:
            accuracy = self.model_performance[symbol].get('accuracy', 0.5)
            confidence_factor = max(0.5, accuracy)
            position_size = int(position_size * confidence_factor)
        
        # Minimum viable size
        min_size = max(1, int(1000 / price))
        
        return max(position_size, min_size)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        model_status = {}
        for symbol, perf in self.model_performance.items():
            model_status[symbol] = {
                'accuracy': perf.get('accuracy', 0),
                'last_trained': perf.get('training_date', 'Never'),
                'samples': perf.get('training_samples', 0)
            }
        
        return {
            'name': 'XGBoost ML Strategy',
            'type': 'Machine Learning - Gradient Boosting',
            'timeframe': f"{self.parameters['prediction_horizon']} day predictions",
            'description': 'Gradient boosting model with comprehensive feature engineering',
            'parameters': self.parameters,
            'risk_level': 'Medium-High',
            'xgboost_available': XGBOOST_AVAILABLE,
            'model_status': model_status,
            'expected_trades_per_day': '2-5',
            'holding_period': f"{self.parameters['prediction_horizon']} days average",
            'requires': ['xgboost', 'scikit-learn'],
            'best_markets': ['Liquid stocks', 'ETFs with rich feature data']
        }