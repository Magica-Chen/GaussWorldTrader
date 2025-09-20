"""
Deep Learning Strategy using Neural Networks

LSTM and CNN-based models for time series prediction and pattern recognition
Incorporates attention mechanisms and ensemble methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    # Create dummy classes for type hints when imports fail
    class keras:
        class Model:
            pass
        class callbacks:
            class History:
                pass
    class MinMaxScaler:
        pass
    class StandardScaler:
        pass

from .base_strategy import BaseStrategy

class DeepLearningStrategy(BaseStrategy):
    """
    Deep Learning trading strategy featuring:
    - LSTM networks for time series forecasting
    - CNN for pattern recognition
    - Attention mechanisms
    - Ensemble of multiple models
    - Advanced feature engineering
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'sequence_length': 60,        # Days of history for LSTM
            'prediction_horizon': 5,      # Days ahead to predict
            'lstm_units': [64, 32],       # LSTM layer sizes
            'cnn_filters': [32, 64],      # CNN filter sizes
            'dense_units': [32, 16],      # Dense layer sizes
            'dropout_rate': 0.2,          # Dropout for regularization
            'learning_rate': 0.001,       # Adam optimizer learning rate
            'batch_size': 32,             # Training batch size
            'epochs': 100,                # Training epochs
            'validation_split': 0.2,      # Validation data split
            'early_stopping_patience': 10, # Early stopping patience
            'min_training_samples': 2000, # Minimum samples for training
            'retrain_frequency': 14,      # Retrain every N days
            'ensemble_models': ['lstm', 'cnn', 'attention'], # Model types
            'confidence_threshold': 0.6,  # Minimum prediction confidence
            'position_size_pct': 0.05,    # 5% of portfolio per position
            'max_positions': 6,           # Maximum concurrent positions
            'stop_loss_pct': 0.025,       # 2.5% stop loss
            'take_profit_pct': 0.05,      # 5% take profit
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available. Install with: pip install tensorflow scikit-learn")
            
        self.models = {}  # Store models per symbol
        self.scalers = {}  # Store scalers per symbol
        self.last_training = {}  # Track last training date per symbol
        self.model_performance = {}  # Track model performance metrics
        
        # Model storage directory
        self.model_dir = Path("models/deep_learning")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure TensorFlow for better performance
        if TENSORFLOW_AVAILABLE:
            tf.config.experimental.enable_op_determinism()
            # Set memory growth for GPU if available
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate deep learning-based trading signals"""
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, skipping deep learning strategy")
            return []
        
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_prices or data.empty:
                continue
            
            try:
                signal = self._analyze_dl_opportunity(
                    symbol, data, current_prices[symbol], 
                    current_data.get(symbol, {}), current_date, portfolio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} with deep learning: {e}")
                continue
        
        return signals
    
    def _analyze_dl_opportunity(self, symbol: str, data: pd.DataFrame,
                               current_price: float, current_bar: Dict[str, Any],
                               current_date: datetime, portfolio: Any = None) -> Optional[Dict[str, Any]]:
        """Analyze deep learning trading opportunity for a single symbol"""
        
        min_data_length = self.parameters['sequence_length'] + self.parameters['prediction_horizon'] + 200
        if len(data) < min_data_length:
            return None
        
        # Check if model needs training/retraining
        needs_training = self._check_training_needed(symbol, current_date)
        
        if needs_training:
            success = self._train_ensemble_models(symbol, data, current_date)
            if not success:
                return None
        
        # Check if models exist
        if symbol not in self.models or not self.models[symbol]:
            return None
        
        # Prepare data for prediction
        prediction_data = self._prepare_prediction_data(symbol, data)
        if prediction_data is None:
            return None
        
        # Make ensemble predictions
        predictions = self._make_ensemble_predictions(symbol, prediction_data)
        if predictions is None:
            return None
        
        prediction, confidence = predictions
        
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
        stop_loss, take_profit = self._calculate_dl_stops(current_price, action, confidence)
        
        return {
            'symbol': symbol,
            'action': action,
            'quantity': position_size,
            'price': current_price,
            'strategy': 'deep_learning',
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'prediction': prediction,
            'model_performance': self.model_performance.get(symbol, {}),
            'holding_period_target': timedelta(days=self.parameters['prediction_horizon']),
            'dl_metadata': {
                'model_types': list(self.models.get(symbol, {}).keys()),
                'ensemble_size': len(self.models.get(symbol, {})),
                'sequence_length': self.parameters['sequence_length'],
                'prediction_horizon': self.parameters['prediction_horizon']
            }
        }
    
    def _check_training_needed(self, symbol: str, current_date: datetime) -> bool:
        """Check if model needs (re)training"""
        
        # No model exists
        if symbol not in self.models or not self.models[symbol]:
            return True
        
        # Check training frequency
        last_train = self.last_training.get(symbol)
        if last_train is None:
            return True
        
        days_since_training = (current_date - last_train).days
        return days_since_training >= self.parameters['retrain_frequency']
    
    def _train_ensemble_models(self, symbol: str, data: pd.DataFrame, current_date: datetime) -> bool:
        """Train ensemble of deep learning models"""
        
        try:
            self.logger.info(f"Training deep learning ensemble for {symbol}")
            
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if len(X) < self.parameters['min_training_samples']:
                self.logger.warning(f"Insufficient training data for {symbol}: {len(X)} samples")
                return False
            
            # Scale data
            scaler = self._fit_scaler(symbol, X)
            X_scaled = scaler.transform(X)
            
            # Reshape for deep learning models
            X_sequences, y_sequences = self._create_sequences(X_scaled, y)
            
            if len(X_sequences) < 100:
                self.logger.warning(f"Insufficient sequences for {symbol}: {len(X_sequences)}")
                return False
            
            # Train ensemble models
            models = {}
            ensemble_predictions = []
            
            for model_type in self.parameters['ensemble_models']:
                try:
                    model = self._create_model(model_type, X_sequences.shape)
                    history = self._train_model(model, X_sequences, y_sequences)
                    
                    # Validate model
                    val_predictions = model.predict(X_sequences[-100:])  # Last 100 samples for validation
                    val_actual = y_sequences[-100:]
                    
                    # Calculate performance
                    mse = mean_squared_error(val_actual, val_predictions.flatten())
                    
                    models[model_type] = {
                        'model': model,
                        'mse': mse,
                        'history': history.history if history else {}
                    }
                    
                    ensemble_predictions.append(val_predictions.flatten())
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type} model for {symbol}: {e}")
                    continue
            
            if not models:
                self.logger.error(f"No models successfully trained for {symbol}")
                return False
            
            # Calculate ensemble performance
            if ensemble_predictions:
                ensemble_pred = np.mean(ensemble_predictions, axis=0)
                ensemble_mse = mean_squared_error(val_actual, ensemble_pred)
            else:
                ensemble_mse = float('inf')
            
            # Store models and metadata
            self.models[symbol] = models
            self.last_training[symbol] = current_date
            self.model_performance[symbol] = {
                'ensemble_mse': ensemble_mse,
                'individual_mse': {k: v['mse'] for k, v in models.items()},
                'training_samples': len(X_sequences),
                'training_date': current_date,
                'model_count': len(models)
            }
            
            # Save models
            self._save_models(symbol)
            
            self.logger.info(f"Deep learning ensemble trained for {symbol} - Models: {len(models)}, MSE: {ensemble_mse:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training deep learning ensemble for {symbol}: {e}")
            return False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        
        # Create comprehensive features
        features = self._engineer_dl_features(data)
        
        # Create targets (future returns)
        horizon = self.parameters['prediction_horizon']
        future_returns = data['close'].pct_change(horizon).shift(-horizon)
        
        # Convert to regression targets (continuous returns)
        targets = future_returns.values
        
        # Align features and targets
        valid_indices = ~(np.isnan(targets) | np.isnan(features).any(axis=1))
        
        features_clean = features[valid_indices]
        targets_clean = targets[valid_indices]
        
        return features_clean, targets_clean
    
    def _engineer_dl_features(self, data: pd.DataFrame) -> np.ndarray:
        """Engineer features optimized for deep learning"""
        
        features_list = []
        
        # Price features (normalized)
        price_features = np.column_stack([
            data['open'].values,
            data['high'].values,
            data['low'].values,
            data['close'].values,
        ])
        features_list.append(price_features)
        
        # Volume features
        volume_features = data['volume'].values.reshape(-1, 1)
        features_list.append(volume_features)
        
        # Technical indicators
        rsi = self._calculate_rsi(data['close'], 14).values.reshape(-1, 1)
        features_list.append(rsi)
        
        # Moving averages
        sma_5 = data['close'].rolling(5).mean().values.reshape(-1, 1)
        sma_20 = data['close'].rolling(20).mean().values.reshape(-1, 1)
        sma_50 = data['close'].rolling(50).mean().values.reshape(-1, 1)
        features_list.append(np.column_stack([sma_5, sma_20, sma_50]))
        
        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(data['close'])
        macd_features = np.column_stack([
            macd_line.values,
            macd_signal.values,
            macd_histogram.values
        ])
        features_list.append(macd_features)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'], 20, 2)
        bb_features = np.column_stack([
            bb_upper.values,
            bb_lower.values,
            bb_middle.values
        ])
        features_list.append(bb_features)
        
        # Returns and volatility
        returns_1d = data['close'].pct_change().values.reshape(-1, 1)
        returns_5d = data['close'].pct_change(5).values.reshape(-1, 1)
        volatility_20d = returns_1d.flatten()
        volatility_20d = pd.Series(volatility_20d).rolling(20).std().values.reshape(-1, 1)
        
        features_list.append(np.column_stack([returns_1d, returns_5d, volatility_20d]))
        
        # Combine all features
        all_features = np.column_stack(features_list)
        
        # Handle NaN values
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return all_features
    
    def _fit_scaler(self, symbol: str, data: np.ndarray) -> MinMaxScaler:
        """Fit and store scaler for the symbol"""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
        self.scalers[symbol] = scaler
        return scaler
    
    def _create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling"""
        
        seq_length = self.parameters['sequence_length']
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_length, len(data)):
            X_sequences.append(data[i-seq_length:i])
            y_sequences.append(targets[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _create_model(self, model_type: str, input_shape: Tuple) -> keras.Model:
        """Create specific model architecture"""
        
        if model_type == 'lstm':
            return self._create_lstm_model(input_shape)
        elif model_type == 'cnn':
            return self._create_cnn_model(input_shape)
        elif model_type == 'attention':
            return self._create_attention_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Create LSTM model for time series prediction"""
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape[1:]))
        
        # LSTM layers
        for i, units in enumerate(self.parameters['lstm_units']):
            return_sequences = i < len(self.parameters['lstm_units']) - 1
            model.add(layers.LSTM(
                units, 
                return_sequences=return_sequences,
                dropout=self.parameters['dropout_rate']
            ))
        
        # Dense layers
        for units in self.parameters['dense_units']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.parameters['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_cnn_model(self, input_shape: Tuple) -> keras.Model:
        """Create CNN model for pattern recognition"""
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape[1:]))
        
        # CNN layers
        for filters in self.parameters['cnn_filters']:
            model.add(layers.Conv1D(
                filters, 
                kernel_size=3, 
                activation='relu',
                padding='same'
            ))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(self.parameters['dropout_rate']))
        
        # Flatten for dense layers
        model.add(layers.GlobalMaxPooling1D())
        
        # Dense layers
        for units in self.parameters['dense_units']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.parameters['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_attention_model(self, input_shape: Tuple) -> keras.Model:
        """Create model with attention mechanism"""
        
        # Input
        inputs = layers.Input(shape=input_shape[1:])
        
        # LSTM with return sequences for attention
        lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        attended = layers.Multiply()([lstm_out, attention])
        attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
        
        # Dense layers
        dense = attended
        for units in self.parameters['dense_units']:
            dense = layers.Dense(units, activation='relu')(dense)
            dense = layers.Dropout(self.parameters['dropout_rate'])(dense)
        
        # Output
        outputs = layers.Dense(1, activation='linear')(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _train_model(self, model: keras.Model, X: np.ndarray, y: np.ndarray) -> Optional[keras.callbacks.History]:
        """Train individual model with callbacks"""
        
        try:
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.parameters['early_stopping_patience'],
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = model.fit(
                X, y,
                batch_size=self.parameters['batch_size'],
                epochs=self.parameters['epochs'],
                validation_split=self.parameters['validation_split'],
                callbacks=callbacks,
                verbose=0
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None
    
    def _prepare_prediction_data(self, symbol: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare data for prediction"""
        
        if symbol not in self.scalers:
            return None
        
        # Engineer features
        features = self._engineer_dl_features(data)
        
        # Scale features
        features_scaled = self.scalers[symbol].transform(features)
        
        # Create sequence for prediction (last sequence)
        seq_length = self.parameters['sequence_length']
        if len(features_scaled) < seq_length:
            return None
        
        prediction_sequence = features_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        return prediction_sequence
    
    def _make_ensemble_predictions(self, symbol: str, prediction_data: np.ndarray) -> Optional[Tuple[float, float]]:
        """Make predictions using ensemble of models"""
        
        if symbol not in self.models or not self.models[symbol]:
            return None
        
        try:
            predictions = []
            weights = []
            
            for model_type, model_info in self.models[symbol].items():
                model = model_info['model']
                mse = model_info['mse']
                
                # Make prediction
                pred = model.predict(prediction_data, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Weight by inverse MSE (better models get higher weight)
                weight = 1.0 / (mse + 1e-8)
                weights.append(weight)
            
            if not predictions:
                return None
            
            # Weighted ensemble prediction
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_prediction = np.average(predictions, weights=weights)
            
            # Calculate confidence based on agreement between models
            prediction_std = np.std(predictions)
            max_std = 0.1  # Maximum expected std for confidence calculation
            confidence = max(0.0, 1.0 - (prediction_std / max_std))
            
            # Convert prediction to probability (0-1 scale)
            # Assuming predictions are returns, convert to probability
            probability = 1.0 / (1.0 + np.exp(-ensemble_prediction * 10))  # Sigmoid scaling
            
            return probability, confidence
            
        except Exception as e:
            self.logger.error(f"Error making ensemble predictions for {symbol}: {e}")
            return None
    
    def _calculate_dl_stops(self, current_price: float, action: str, confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on DL confidence"""
        
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
        """Check portfolio constraints for DL strategy"""
        if portfolio is None:
            return True
        
        # Check maximum positions
        current_positions = len([pos for pos in portfolio.positions.values() 
                               if pos['quantity'] != 0])
        
        if current_positions >= self.parameters['max_positions']:
            return False
        
        return True
    
    def _save_models(self, symbol: str) -> None:
        """Save trained models to disk"""
        try:
            symbol_dir = self.model_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Save each model in the ensemble
            for model_type, model_info in self.models[symbol].items():
                model_path = symbol_dir / f"{model_type}_model"
                model_info['model'].save(model_path)
            
            # Save scaler
            if symbol in self.scalers:
                scaler_path = symbol_dir / "scaler.joblib"
                joblib.dump(self.scalers[symbol], scaler_path)
                
        except Exception as e:
            self.logger.error(f"Error saving models for {symbol}: {e}")
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for DL strategy"""
        
        # Base position size
        target_value = portfolio_value * self.parameters['position_size_pct']
        position_size = int(target_value / price)
        
        # Adjust based on model performance if available
        if symbol in self.model_performance:
            ensemble_mse = self.model_performance[symbol].get('ensemble_mse', 1.0)
            performance_factor = max(0.3, 1.0 / (1.0 + ensemble_mse * 100))
            position_size = int(position_size * performance_factor)
        
        # Minimum viable size
        min_size = max(1, int(1000 / price))
        
        return max(position_size, min_size)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        model_status = {}
        for symbol, perf in self.model_performance.items():
            model_status[symbol] = {
                'ensemble_mse': perf.get('ensemble_mse', 0),
                'model_count': perf.get('model_count', 0),
                'last_trained': perf.get('training_date', 'Never'),
                'samples': perf.get('training_samples', 0)
            }
        
        return {
            'name': 'Deep Learning Strategy',
            'type': 'Machine Learning - Neural Networks',
            'timeframe': f"{self.parameters['prediction_horizon']} day predictions",
            'description': 'Ensemble of LSTM, CNN, and Attention models for time series prediction',
            'parameters': self.parameters,
            'risk_level': 'Medium-High',
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'model_status': model_status,
            'ensemble_types': self.parameters['ensemble_models'],
            'expected_trades_per_day': '1-3',
            'holding_period': f"{self.parameters['prediction_horizon']} days average",
            'requires': ['tensorflow', 'scikit-learn'],
            'best_markets': ['Liquid stocks', 'ETFs with long history']
        }