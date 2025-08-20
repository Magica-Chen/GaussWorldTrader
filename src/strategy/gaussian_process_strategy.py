"""
Gaussian Process Strategy

Bayesian approach to trading using Gaussian Processes for uncertainty quantification
Provides probabilistic predictions with confidence intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import scipy.stats as stats
    GAUSSIAN_PROCESS_AVAILABLE = True
except ImportError:
    GAUSSIAN_PROCESS_AVAILABLE = False
    # Create dummy classes for type hints when imports fail
    class StandardScaler:
        pass

from .base_strategy import BaseStrategy

class GaussianProcessStrategy(BaseStrategy):
    """
    Gaussian Process trading strategy featuring:
    - Bayesian probabilistic predictions
    - Uncertainty quantification
    - Adaptive kernel learning
    - Risk-aware position sizing based on prediction uncertainty
    - Multiple kernel combinations
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'prediction_horizon': 5,      # Days ahead to predict
            'feature_lookback': 45,       # Days of features to use
            'min_training_samples': 500,  # Minimum samples for training
            'retrain_frequency': 21,      # Retrain every N days
            'uncertainty_threshold': 0.02, # Maximum uncertainty for trading
            'confidence_level': 0.68,     # 1-sigma confidence level
            'position_size_base_pct': 0.04, # Base 4% of portfolio per position
            'max_positions': 6,           # Maximum concurrent positions
            'kernel_type': 'adaptive',    # 'rbf', 'matern', 'adaptive'
            'alpha': 1e-6,               # Noise level in GP
            'n_restarts_optimizer': 5,    # GP hyperparameter optimization
            'normalize_y': True,          # Normalize targets
            'acquisition_function': 'ei', # 'ei', 'ucb', 'pi' for exploration
            'ucb_beta': 2.0,             # UCB acquisition parameter
            'kelly_fraction': False,      # Use Kelly criterion for sizing
            'risk_free_rate': 0.02       # Risk-free rate for Sharpe calculations
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(default_params)
        self.logger = logging.getLogger(__name__)
        
        if not GAUSSIAN_PROCESS_AVAILABLE:
            self.logger.error("Gaussian Process libraries not available. Install with: pip install scikit-learn scipy")
            
        self.models = {}  # Store GP models per symbol
        self.scalers = {}  # Store scalers per symbol
        self.last_training = {}  # Track last training date per symbol
        self.model_performance = {}  # Track model performance metrics
        self.prediction_history = {}  # Store prediction history for calibration
        
        # Model storage directory
        self.model_dir = Path("models/gaussian_process")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_signals(self, current_date: datetime, current_prices: Dict[str, float],
                        current_data: Dict[str, Any], historical_data: Dict[str, pd.DataFrame],
                        portfolio: Any = None) -> List[Dict[str, Any]]:
        """Generate Gaussian Process-based trading signals"""
        
        if not GAUSSIAN_PROCESS_AVAILABLE:
            self.logger.warning("Gaussian Process libraries not available, skipping GP strategy")
            return []
        
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_prices or data.empty:
                continue
            
            try:
                signal = self._analyze_gp_opportunity(
                    symbol, data, current_prices[symbol], 
                    current_data.get(symbol, {}), current_date, portfolio
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} with Gaussian Process: {e}")
                continue
        
        return signals
    
    def _analyze_gp_opportunity(self, symbol: str, data: pd.DataFrame,
                               current_price: float, current_bar: Dict[str, Any],
                               current_date: datetime, portfolio: Any = None) -> Optional[Dict[str, Any]]:
        """Analyze GP trading opportunity for a single symbol"""
        
        min_data_length = self.parameters['feature_lookback'] + self.parameters['prediction_horizon'] + 100
        if len(data) < min_data_length:
            return None
        
        # Check if model needs training/retraining
        needs_training = self._check_training_needed(symbol, current_date)
        
        if needs_training:
            success = self._train_gp_model(symbol, data, current_date)
            if not success:
                return None
        
        # Check if model exists
        if symbol not in self.models:
            return None
        
        # Prepare features for prediction
        features = self._engineer_gp_features(data, for_prediction=True)
        if features is None or len(features) == 0:
            return None
        
        # Make probabilistic prediction
        prediction_result = self._make_probabilistic_prediction(symbol, features)
        if prediction_result is None:
            return None
        
        mean_pred, std_pred, confidence_interval = prediction_result
        
        # Check uncertainty threshold
        uncertainty = std_pred / abs(mean_pred) if abs(mean_pred) > 1e-6 else float('inf')
        if uncertainty > self.parameters['uncertainty_threshold']:
            return None
        
        # Determine action using acquisition function
        action_result = self._determine_action_with_acquisition(mean_pred, std_pred, confidence_interval)
        if action_result is None:
            return None
        
        action, confidence = action_result
        
        # Portfolio constraints
        if not self._check_portfolio_constraints(symbol, action, portfolio):
            return None
        
        # Calculate position size based on uncertainty
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price}) if portfolio else 100000
        position_size = self._calculate_uncertainty_based_position_size(
            symbol, current_price, portfolio_value, std_pred, mean_pred
        )
        
        # Risk management with uncertainty
        stop_loss, take_profit = self._calculate_gp_stops(current_price, action, std_pred, mean_pred)
        
        return {
            'symbol': symbol,
            'action': action,
            'quantity': position_size,
            'price': current_price,
            'strategy': 'gaussian_process',
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'prediction_mean': mean_pred,
            'prediction_std': std_pred,
            'uncertainty': uncertainty,
            'confidence_interval': confidence_interval,
            'model_performance': self.model_performance.get(symbol, {}),
            'holding_period_target': timedelta(days=self.parameters['prediction_horizon']),
            'gp_metadata': {
                'kernel_type': self.parameters['kernel_type'],
                'uncertainty_threshold': self.parameters['uncertainty_threshold'],
                'acquisition_function': self.parameters['acquisition_function']
            }
        }
    
    def _check_training_needed(self, symbol: str, current_date: datetime) -> bool:
        """Check if GP model needs (re)training"""
        
        # No model exists
        if symbol not in self.models:
            return True
        
        # Check training frequency
        last_train = self.last_training.get(symbol)
        if last_train is None:
            return True
        
        days_since_training = (current_date - last_train).days
        return days_since_training >= self.parameters['retrain_frequency']
    
    def _train_gp_model(self, symbol: str, data: pd.DataFrame, current_date: datetime) -> bool:
        """Train Gaussian Process model for the symbol"""
        
        try:
            self.logger.info(f"Training Gaussian Process model for {symbol}")
            
            # Prepare training data
            features, targets = self._prepare_gp_training_data(data)
            
            if len(features) < self.parameters['min_training_samples']:
                self.logger.warning(f"Insufficient training data for {symbol}: {len(features)} samples")
                return False
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers[symbol] = scaler
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets, test_size=0.2, random_state=42
            )
            
            # Create and train GP model
            kernel = self._create_kernel()
            
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.parameters['alpha'],
                n_restarts_optimizer=self.parameters['n_restarts_optimizer'],
                normalize_y=self.parameters['normalize_y'],
                random_state=42
            )
            
            # Fit the model
            gp_model.fit(X_train, y_train)
            
            # Validate model
            val_mean, val_std = gp_model.predict(X_val, return_std=True)
            
            # Calculate performance metrics
            mse = mean_squared_error(y_val, val_mean)
            mae = mean_absolute_error(y_val, val_mean)
            
            # Calculate prediction intervals coverage
            confidence_level = self.parameters['confidence_level']
            z_score = stats.norm.ppf(0.5 + confidence_level / 2)
            
            lower_bound = val_mean - z_score * val_std
            upper_bound = val_mean + z_score * val_std
            
            coverage = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
            
            # Calculate log marginal likelihood (model evidence)
            log_likelihood = gp_model.log_marginal_likelihood()
            
            # Store model and performance
            self.models[symbol] = gp_model
            self.last_training[symbol] = current_date
            self.model_performance[symbol] = {
                'mse': mse,
                'mae': mae,
                'coverage': coverage,
                'expected_coverage': confidence_level,
                'log_likelihood': log_likelihood,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'training_date': current_date,
                'kernel_params': gp_model.kernel_.get_params()
            }
            
            # Save model
            self._save_gp_model(symbol, gp_model)
            
            self.logger.info(f"GP model trained for {symbol} - MSE: {mse:.6f}, Coverage: {coverage:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training GP model for {symbol}: {e}")
            return False
    
    def _prepare_gp_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for GP training"""
        
        # Generate features
        features_df = self._engineer_gp_features(data, for_prediction=False)
        
        # Create targets (future returns)
        horizon = self.parameters['prediction_horizon']
        future_returns = data['close'].pct_change(horizon).shift(-horizon)
        
        # Align features and targets
        valid_indices = ~(np.isnan(future_returns) | features_df.isnull().any(axis=1))
        
        features_clean = features_df[valid_indices].values
        targets_clean = future_returns[valid_indices].values
        
        return features_clean, targets_clean
    
    def _engineer_gp_features(self, data: pd.DataFrame, for_prediction: bool = False) -> Optional[pd.DataFrame]:
        """Engineer features optimized for Gaussian Process"""
        
        if len(data) < self.parameters['feature_lookback']:
            return None
        
        features = pd.DataFrame(index=data.index)
        
        # Returns at multiple horizons
        features['returns_1d'] = data['close'].pct_change(1)
        features['returns_3d'] = data['close'].pct_change(3)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_10d'] = data['close'].pct_change(10)
        
        # Volatility measures
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['rsi_7'] = self._calculate_rsi(data['close'], 7)
        
        # Moving averages ratios
        sma_5 = data['close'].rolling(5).mean()
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        features['price_to_sma5'] = data['close'] / sma_5
        features['price_to_sma20'] = data['close'] / sma_20
        features['sma5_to_sma20'] = sma_5 / sma_20
        features['sma20_to_sma50'] = sma_20 / sma_50
        
        # MACD features
        macd_line, macd_signal, macd_histogram = self._calculate_macd(data['close'])
        features['macd_line'] = macd_line
        features['macd_histogram'] = macd_histogram
        
        # Bollinger Bands position
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'], 20, 2)
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['price_volume'] = data['close'] * data['volume']
        
        # Market microstructure
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Momentum features
        features['momentum_12'] = data['close'].pct_change(12)
        features['momentum_26'] = data['close'].pct_change(26)
        
        # Regime indicators
        features['volatility_regime'] = np.where(
            features['volatility_20d'] > features['volatility_20d'].rolling(60).mean(), 1, 0
        )
        
        # Lagged features (important for GP temporal modeling)
        for lag in [1, 2, 3]:
            features[f'returns_1d_lag{lag}'] = features['returns_1d'].shift(lag)
            features[f'rsi_lag{lag}'] = features['rsi_14'].shift(lag)
            features[f'volatility_lag{lag}'] = features['volatility_5d'].shift(lag)
        
        # Clean data
        features = features.fillna(method='ffill').fillna(0)
        
        # For prediction, return only the last row
        if for_prediction:
            if len(features) > 0:
                return features.iloc[-1:].values
            else:
                return None
        
        return features
    
    def _create_kernel(self) -> Any:
        """Create kernel for Gaussian Process"""
        
        kernel_type = self.parameters['kernel_type']
        
        if kernel_type == 'rbf':
            # RBF kernel with white noise
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
            
        elif kernel_type == 'matern':
            # Matérn kernel (less smooth than RBF)
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3)
            
        elif kernel_type == 'adaptive':
            # Composite kernel that can capture multiple patterns
            kernel = (ConstantKernel(1.0) * RBF(1.0) + 
                     ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) +
                     WhiteKernel(1e-3))
        
        else:
            # Default to RBF
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        
        return kernel
    
    def _make_probabilistic_prediction(self, symbol: str, features: np.ndarray) -> Optional[Tuple[float, float, Tuple[float, float]]]:
        """Make probabilistic prediction with uncertainty"""
        
        if symbol not in self.models or symbol not in self.scalers:
            return None
        
        try:
            # Scale features
            scaler = self.scalers[symbol]
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Make prediction with uncertainty
            gp_model = self.models[symbol]
            mean_pred, std_pred = gp_model.predict(features_scaled, return_std=True)
            
            mean_pred = mean_pred[0]
            std_pred = std_pred[0]
            
            # Calculate confidence interval
            confidence_level = self.parameters['confidence_level']
            z_score = stats.norm.ppf(0.5 + confidence_level / 2)
            
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
            confidence_interval = (lower_bound, upper_bound)
            
            return mean_pred, std_pred, confidence_interval
            
        except Exception as e:
            self.logger.error(f"Error making GP prediction for {symbol}: {e}")
            return None
    
    def _determine_action_with_acquisition(self, mean_pred: float, std_pred: float, 
                                          confidence_interval: Tuple[float, float]) -> Optional[Tuple[str, float]]:
        """Determine action using acquisition function"""
        
        acquisition_func = self.parameters['acquisition_function']
        
        if acquisition_func == 'ei':  # Expected Improvement
            # Use mean prediction with uncertainty bonus
            threshold = 0.005  # 0.5% threshold for action
            
            if mean_pred > threshold:
                # Probability of improvement
                z = (mean_pred - threshold) / (std_pred + 1e-8)
                prob_improvement = stats.norm.cdf(z)
                expected_improvement = std_pred * stats.norm.pdf(z) + (mean_pred - threshold) * prob_improvement
                
                if expected_improvement > 0.001:  # Minimum EI threshold
                    return 'buy', min(prob_improvement, 1.0)
            
            elif mean_pred < -threshold:
                # Probability of negative improvement (for selling)
                z = (-mean_pred - threshold) / (std_pred + 1e-8)
                prob_improvement = stats.norm.cdf(z)
                expected_improvement = std_pred * stats.norm.pdf(z) + (-mean_pred - threshold) * prob_improvement
                
                if expected_improvement > 0.001:
                    return 'sell', min(prob_improvement, 1.0)
        
        elif acquisition_func == 'ucb':  # Upper Confidence Bound
            beta = self.parameters['ucb_beta']
            
            # UCB for buying (upper bound of confidence interval)
            ucb = mean_pred + beta * std_pred
            
            if ucb > 0.01:  # 1% threshold
                confidence = min((ucb - 0.01) / 0.05, 1.0)  # Scale confidence
                return 'buy', confidence
            
            # LCB for selling (lower bound)
            lcb = mean_pred - beta * std_pred
            
            if lcb < -0.01:
                confidence = min((-lcb - 0.01) / 0.05, 1.0)
                return 'sell', confidence
        
        elif acquisition_func == 'pi':  # Probability of Improvement
            threshold = 0.005
            
            if std_pred > 0:
                # Probability that prediction exceeds threshold
                z_buy = (mean_pred - threshold) / std_pred
                prob_buy = stats.norm.cdf(z_buy)
                
                z_sell = (-mean_pred - threshold) / std_pred
                prob_sell = stats.norm.cdf(z_sell)
                
                if prob_buy > 0.6:
                    return 'buy', prob_buy
                elif prob_sell > 0.6:
                    return 'sell', prob_sell
        
        return None
    
    def _calculate_uncertainty_based_position_size(self, symbol: str, price: float, 
                                                  portfolio_value: float, prediction_std: float,
                                                  prediction_mean: float) -> int:
        """Calculate position size considering prediction uncertainty"""
        
        base_size_pct = self.parameters['position_size_base_pct']
        
        # Adjust size based on uncertainty (lower uncertainty = larger position)
        uncertainty = prediction_std / abs(prediction_mean) if abs(prediction_mean) > 1e-6 else 1.0
        uncertainty_factor = max(0.1, 1.0 - uncertainty)  # Reduce size for high uncertainty
        
        # Kelly criterion adjustment if enabled
        if self.parameters['kelly_fraction'] and abs(prediction_mean) > 1e-6:
            # Simplified Kelly fraction: f = (bp - q) / b
            # where b = odds, p = probability of win, q = probability of loss
            
            # Estimate win probability from prediction
            prob_win = stats.norm.cdf(prediction_mean / prediction_std) if prediction_std > 0 else 0.5
            
            # Estimate expected return and variance
            expected_return = abs(prediction_mean)
            variance = prediction_std ** 2
            
            if variance > 0 and expected_return > 0:
                kelly_fraction = expected_return / variance
                kelly_fraction = max(0.01, min(kelly_fraction, 0.25))  # Cap at 25%
                
                size_pct = min(base_size_pct, kelly_fraction * uncertainty_factor)
            else:
                size_pct = base_size_pct * uncertainty_factor
        else:
            size_pct = base_size_pct * uncertainty_factor
        
        # Calculate position size
        target_value = portfolio_value * size_pct
        position_size = int(target_value / price)
        
        # Minimum viable size
        min_size = max(1, int(500 / price))
        
        return max(position_size, min_size)
    
    def _calculate_gp_stops(self, current_price: float, action: str, 
                           prediction_std: float, prediction_mean: float) -> Tuple[float, float]:
        """Calculate stops based on prediction uncertainty"""
        
        # Use prediction uncertainty to set wider stops for uncertain predictions
        base_stop_pct = 0.02  # 2% base stop
        base_profit_pct = 0.04  # 4% base profit
        
        # Adjust based on prediction standard deviation
        uncertainty_multiplier = max(1.0, 1.0 + prediction_std * 10)  # Scale uncertainty
        
        stop_pct = base_stop_pct * uncertainty_multiplier
        profit_pct = base_profit_pct * uncertainty_multiplier
        
        if action == 'buy':
            stop_loss = current_price * (1 - stop_pct)
            take_profit = current_price * (1 + profit_pct)
        else:  # sell
            stop_loss = current_price * (1 + stop_pct)
            take_profit = current_price * (1 - profit_pct)
        
        return stop_loss, take_profit
    
    def _check_portfolio_constraints(self, symbol: str, action: str, portfolio: Any) -> bool:
        """Check portfolio constraints for GP strategy"""
        if portfolio is None:
            return True
        
        # Check maximum positions
        current_positions = len([pos for pos in portfolio.positions.values() 
                               if pos['quantity'] != 0])
        
        if current_positions >= self.parameters['max_positions']:
            return False
        
        return True
    
    def _save_gp_model(self, symbol: str, model) -> None:
        """Save GP model to disk"""
        try:
            model_path = self.model_dir / f"{symbol}_gp_model.joblib"
            scaler_path = self.model_dir / f"{symbol}_gp_scaler.joblib"
            
            joblib.dump(model, model_path)
            if symbol in self.scalers:
                joblib.dump(self.scalers[symbol], scaler_path)
                
        except Exception as e:
            self.logger.error(f"Error saving GP model for {symbol}: {e}")
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = None) -> int:
        """Calculate position size for GP strategy"""
        
        # Use uncertainty-based sizing if model is available
        if symbol in self.models:
            # This would be called from the main analysis function
            # For now, use base sizing
            pass
        
        # Base position size
        target_value = portfolio_value * self.parameters['position_size_base_pct']
        position_size = int(target_value / price)
        
        # Minimum viable size
        min_size = max(1, int(500 / price))
        
        return max(position_size, min_size)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        model_status = {}
        for symbol, perf in self.model_performance.items():
            model_status[symbol] = {
                'mse': perf.get('mse', 0),
                'coverage': perf.get('coverage', 0),
                'expected_coverage': perf.get('expected_coverage', 0),
                'log_likelihood': perf.get('log_likelihood', 0),
                'last_trained': perf.get('training_date', 'Never'),
                'samples': perf.get('training_samples', 0)
            }
        
        return {
            'name': 'Gaussian Process Strategy',
            'type': 'Bayesian Machine Learning',
            'timeframe': f"{self.parameters['prediction_horizon']} day predictions",
            'description': 'Bayesian approach with uncertainty quantification and probabilistic predictions',
            'parameters': self.parameters,
            'risk_level': 'Medium',
            'gaussian_process_available': GAUSSIAN_PROCESS_AVAILABLE,
            'model_status': model_status,
            'expected_trades_per_day': '1-2',
            'holding_period': f"{self.parameters['prediction_horizon']} days average",
            'requires': ['scikit-learn', 'scipy'],
            'best_markets': ['Any liquid market', 'Works well with noisy data'],
            'advantages': [
                'Uncertainty quantification',
                'Probabilistic predictions',
                'Robust to overfitting',
                'Adaptive position sizing'
            ]
        }