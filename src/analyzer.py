import pandas as pd
import pandas_ta as ta
from typing import Dict, List
from datetime import datetime

class PriceAnalyzer:
    LOOKBACK_REQUIREMENTS = {
        'sma_200': 200,
        'ichimoku': {
            'lookback': 52,
            'forward': 26
        },
        'macd': 26,
        'rsi': 14,
        'bbands': 20
    }

    @staticmethod
    def analyze_for_algorithm(data: List[Dict]) -> List[Dict]:
        """
        Analyzes price data for algorithmic trading/backtesting
        """
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data)
        
        if len(df) < PriceAnalyzer.LOOKBACK_REQUIREMENTS['sma_200']:
            raise ValueError(f"Need at least 200 periods of data, got {len(df)}")

        df = df.sort_values('timestamp')
        df.ta.strategy(
            ta.Strategy(
                name="crypto_analysis",
                ta=[
                    {"kind": "sma", "length": 200},
                    {"kind": "ichimoku"},
                    {"kind": "rsi"},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "bbands", "length": 20}
                ]
            )
        )

        analyzed_data = []
        ichimoku_future = PriceAnalyzer.LOOKBACK_REQUIREMENTS['ichimoku']['forward']
        
        for idx in range(200, len(df) - ichimoku_future):
            row = df.iloc[idx]
            future_row = df.iloc[idx + ichimoku_future]

            indicators = {
                'trend_indicators': {
                    'sma_200': float(row.get('SMA_200', 0)),
                    'ichimoku': {
                        'tenkan': float(row.get('ICHIMOKU_tenkan_sen', 0)),
                        'kijun': float(row.get('ICHIMOKU_kijun_sen', 0)),
                        'senkou_a': float(future_row.get('ICHIMOKU_senkou_span_a', 0)),
                        'senkou_b': float(future_row.get('ICHIMOKU_senkou_span_b', 0)),
                        'chikou': float(row.get('ICHIMOKU_chikou_span', 0))
                    }
                },
                'momentum_indicators': {
                    'rsi': float(row.get('RSI_14', 0)),
                    'macd': {
                        'line': float(row.get('MACD_12_26_9', 0)),
                        'signal': float(row.get('MACDs_12_26_9', 0)),
                        'histogram': float(row.get('MACDh_12_26_9', 0))
                    }
                },
                'volatility_indicators': {
                    'bollinger_bands': {
                        'upper': float(row.get('BBU_20_2.0', 0)),
                        'middle': float(row.get('BBM_20_2.0', 0)),
                        'lower': float(row.get('BBL_20_2.0', 0))
                    }
                }
            }

            analyzed_data.append({
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                **indicators
            })

        return analyzed_data