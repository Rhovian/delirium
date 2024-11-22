"""
analyze.py - Quantitative analysis and backtesting system
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from hmmlearn import hmm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class MarketRegime(Enum):
    BULLISH = 1
    BEARISH = 2
    SIDEWAYS = 3

@dataclass
class MarketState:
    regime: MarketRegime
    volatility: float
    trend_strength: float

@dataclass
class Position:
    entry_price: float
    size: float
    entry_time: datetime
    direction: int  # 1 for long, -1 for short
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None

@dataclass
class BacktestStats:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    profitable_trades: int
    positions: List[Position]
    equity_curve: List[float]

class QuantAnalyzer:
    def __init__(self, lookback_period: int = 200):
        self.lookback_period = lookback_period
        self.hmm_model = self._initialize_hmm()
        
    def _initialize_hmm(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            init_params=""
        )
    
    def prepare_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        returns = np.log(df['close']).diff()
        volatility = returns.rolling(window=5).std()
        rsi = ta.rsi(df['close'], length=14)
        sma_ratio = df['close'] / ta.sma(df['close'], length=200)
        
        features = np.column_stack([
            returns.fillna(0) / returns.std(),
            volatility.fillna(0) / volatility.std(),
            (rsi.fillna(50) - 50) / 25,
            np.log(sma_ratio.fillna(1))
        ])
        
        return features[200:]

    def map_hmm_states_to_regimes(self, states: np.ndarray, prices: pd.Series) -> List[MarketRegime]:
        returns_by_state = {
            state: prices[states == state].pct_change().mean()
            for state in range(3)
        }
        
        sorted_states = sorted(returns_by_state.items(), key=lambda x: x[1])
        state_mapping = {
            sorted_states[0][0]: MarketRegime.BEARISH,
            sorted_states[1][0]: MarketRegime.SIDEWAYS,
            sorted_states[2][0]: MarketRegime.BULLISH
        }
        
        return [state_mapping[state] for state in states]
        
    def process_timepoint(self, df: pd.DataFrame, idx: int, regime_mapping: List[MarketRegime]) -> Dict:
        row = df.iloc[idx]
        regime_idx = idx - 200
        
        state = MarketState(
            regime=regime_mapping[regime_idx] if regime_idx >= 0 else MarketRegime.SIDEWAYS,
            volatility=float(row.get('ATR_14', 0)),
            trend_strength=abs(float(row.get('MACD_12_26_9', 0)))
        )
        
        indicators = {
            'trend_indicators': {
                'sma_200': float(row.get('SMA_200', 0)),
                'price_to_sma': float(row['close'] / row.get('SMA_200', row['close'])),
                'macd': {
                    'line': float(row.get('MACD_12_26_9', 0)),
                    'signal': float(row.get('MACDs_12_26_9', 0)),
                    'histogram': float(row.get('MACDh_12_26_9', 0))
                }
            },
            'momentum_indicators': {
                'rsi': float(row.get('RSI_14', 0))
            },
            'volatility_indicators': {
                'atr': float(row.get('ATR_14', 0)),
                'bollinger_bands': {
                    'upper': float(row.get('BBU_20_2.0', 0)),
                    'middle': float(row.get('BBM_20_2.0', 0)),
                    'lower': float(row.get('BBL_20_2.0', 0))
                }
            },
            'market_state': {
                'regime': state.regime.name,
                'trend_strength': state.trend_strength
            }
        }
        
        return {
            'timestamp': row['timestamp'],
            'symbol': row['symbol'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            **indicators
        }
        
    def analyze_data(self, df: pd.DataFrame) -> List[Dict]:
        if len(df) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} periods of data")
            
        df = df.copy().sort_values('timestamp')
        
        # Calculate technical indicators
        df.ta.strategy(
            ta.Strategy(
                name="hybrid_analysis",
                ta=[
                    {"kind": "sma", "length": 200},
                    {"kind": "rsi"},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "bbands", "length": 20},
                    {"kind": "atr", "length": 14}
                ]
            )
        )
        
        # Prepare and fit HMM
        hmm_features = self.prepare_hmm_features(df)
        self.hmm_model.fit(hmm_features)
        regime_states = self.hmm_model.predict(hmm_features)
        regime_mapping = self.map_hmm_states_to_regimes(regime_states, df['close'].iloc[200:])
        
        analyzed_data = []
        
        for idx in range(200, len(df)):
            try:
                analyzed_data.append(self.process_timepoint(df, idx, regime_mapping))
            except Exception as e:
                print(f"Warning: Failed to process data point at {df.iloc[idx]['timestamp']}: {str(e)}")
                continue
                
        if not analyzed_data:
            raise ValueError("No valid analyzed data points produced")
            
        return analyzed_data

class SignalGenerator:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        
    def generate_signal(self, data: Dict) -> float:
        signals = {
            'trend': self._analyze_trend(data),
            'momentum': self._analyze_momentum(data),
            'volatility': self._analyze_volatility(data),
            'regime': self._analyze_regime(data)
        }
        
        final_signal = sum(
            self.weights.get(key, 0) * value 
            for key, value in signals.items()
        )
        
        return np.clip(final_signal, -1, 1)
    
    def _analyze_trend(self, data: Dict) -> float:
        trend = data['trend_indicators']
        price_to_sma = trend['price_to_sma']
        macd = trend['macd']['histogram']
        
        trend_signal = (
            (price_to_sma - 1) +
            np.tanh(macd)
        ) / 2
        
        return np.clip(trend_signal, -1, 1)
    
    def _analyze_momentum(self, data: Dict) -> float:
        rsi = data['momentum_indicators']['rsi']
        rsi_signal = (rsi - 50) / 50
        
        if rsi < 30:
            rsi_signal *= 0.5
        elif rsi > 70:
            rsi_signal *= 0.5
            
        return rsi_signal
    
    def _analyze_volatility(self, data: Dict) -> float:
        bb = data['volatility_indicators']['bollinger_bands']
        current_price = data['close']
        
        bb_range = bb['upper'] - bb['lower']
        if bb_range == 0:
            return 0
            
        bb_position = (current_price - bb['lower']) / bb_range
        return (bb_position - 0.5) * 2
    
    def _analyze_regime(self, data: Dict) -> float:
        regime = data['market_state']['regime']
        trend_strength = data['market_state']['trend_strength']
        
        base_signal = {
            'BULLISH': 1,
            'BEARISH': -1,
            'SIDEWAYS': 0
        }.get(regime, 0)
        
        return base_signal * min(1, trend_strength)

class BacktestEngine:
    def __init__(self, analyzer: QuantAnalyzer, signal_generator: SignalGenerator):
        self.analyzer = analyzer
        self.signal_generator = signal_generator
        self.positions: List[Position] = []
        self.current_position: Optional[Position] = None
    
    def run(self, df: pd.DataFrame, initial_capital: float, risk_per_trade: float = 0.02) -> BacktestStats:
        analyzed_data = self.analyzer.analyze_data(df)
        capital = initial_capital
        equity_curve = [initial_capital]
        
        for i in range(len(analyzed_data)):
            data = analyzed_data[i]
            signal = self.signal_generator.generate_signal(data)
            
            # Position sizing
            position_size = self._calculate_position_size(
                capital, 
                risk_per_trade,
                data['volatility_indicators']['atr'],
                data['close']
            )
            
            # Check for exit if we have a position
            if self.current_position:
                if (self.current_position.direction == 1 and signal < -0.2) or \
                   (self.current_position.direction == -1 and signal > 0.2):
                    self._close_position(data['timestamp'], data['close'])
                    capital += self.positions[-1].pnl
            
            # Check for entry
            if not self.current_position:
                if signal > 0.5:
                    self._open_position(data['timestamp'], data['close'], position_size, 1)
                elif signal < -0.5:
                    self._open_position(data['timestamp'], data['close'], position_size, -1)
            
            # Update equity
            if self.current_position:
                unrealized_pnl = (
                    (data['close'] - self.current_position.entry_price)
                    * self.current_position.size
                    * self.current_position.direction
                )
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Close any remaining position
        if self.current_position:
            self._close_position(analyzed_data[-1]['timestamp'], analyzed_data[-1]['close'])
            capital += self.positions[-1].pnl
        
        # Calculate statistics
        returns = pd.Series(equity_curve).pct_change().dropna()
        profitable_trades = len([p for p in self.positions if p.pnl and p.pnl > 0])
        total_trades = len(self.positions)
        
        if total_trades > 0:
            win_rate = profitable_trades / total_trades
            profits = [p.pnl for p in self.positions if p.pnl and p.pnl > 0]
            losses = [p.pnl for p in self.positions if p.pnl and p.pnl <= 0]
            
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        return BacktestStats(
            total_return=(equity_curve[-1] - initial_capital) / initial_capital,
            sharpe_ratio=np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            positions=self.positions,
            equity_curve=equity_curve
        )
    
    def _calculate_position_size(self, capital: float, risk_per_trade: float,
                               volatility: float, price: float) -> float:
        risk_amount = capital * risk_per_trade
        volatility_factor = 1 / (1 + volatility)
        return (risk_amount * volatility_factor) / price
    
    def _open_position(self, timestamp: datetime, price: float, size: float, direction: int):
        self.current_position = Position(
            entry_price=price,
            size=size,
            entry_time=timestamp,
            direction=direction
        )
    
    def _close_position(self, timestamp: datetime, price: float):
        if self.current_position:
            self.current_position.exit_price = price
            self.current_position.exit_time = timestamp
            self.current_position.pnl = (
                (price - self.current_position.entry_price)
                * self.current_position.size
                * self.current_position.direction
            )
            self.positions.append(self.current_position)
            self.current_position = None
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd