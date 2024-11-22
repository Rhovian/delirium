"""
visualization.py - Enhanced visualization with HMM states and error handling
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

class TradingSystemVisualizer:
    @staticmethod
    def create_dashboard(df: pd.DataFrame, analyzed_data: List[Dict], backtest_stats: 'BacktestStats') -> go.Figure:
        """Create an interactive dashboard showing trading system performance"""
        try:
            # Convert analyzed data to DataFrame for easier plotting
            analysis_df = pd.DataFrame(analyzed_data)
            
            fig = make_subplots(
                rows=5, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    'Price & Trades',
                    'Market Regime',
                    'RSI',
                    'MACD',
                    'Equity Curve'
                ),
                row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
            )
            
            # 1. Price chart with trades
            fig.add_trace(
                go.Candlestick(
                    x=analysis_df['timestamp'],
                    open=analysis_df['open'],
                    high=analysis_df['high'],
                    low=analysis_df['low'],
                    close=analysis_df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add SMA
            fig.add_trace(
                go.Scatter(
                    x=analysis_df['timestamp'],
                    y=[d['trend_indicators']['sma_200'] for d in analyzed_data],
                    name='SMA 200',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            # Add trade markers
            TradingSystemVisualizer._add_trade_markers(fig, backtest_stats.positions)
            
            # 2. Market Regime
            regimes = [d['market_state']['regime'] for d in analyzed_data]
            timestamps = analysis_df['timestamp']
            
            # Create separate traces for each regime type
            bullish_mask = [r == 'BULLISH' for r in regimes]
            bearish_mask = [r == 'BEARISH' for r in regimes]
            sideways_mask = [r == 'SIDEWAYS' for r in regimes]
            
            # Plot bullish periods
            if any(bullish_mask):
                fig.add_trace(
                    go.Scatter(
                        x=[t for t, m in zip(timestamps, bullish_mask) if m],
                        y=[1 for m in bullish_mask if m],
                        name='Bullish',
                        mode='lines',
                        line=dict(color='green', width=2),
                        connectgaps=False
                    ),
                    row=2, col=1
                )
            
            # Plot bearish periods
            if any(bearish_mask):
                fig.add_trace(
                    go.Scatter(
                        x=[t for t, m in zip(timestamps, bearish_mask) if m],
                        y=[-1 for m in bearish_mask if m],
                        name='Bearish',
                        mode='lines',
                        line=dict(color='red', width=2),
                        connectgaps=False
                    ),
                    row=2, col=1
                )
            
            # Plot sideways periods
            if any(sideways_mask):
                fig.add_trace(
                    go.Scatter(
                        x=[t for t, m in zip(timestamps, sideways_mask) if m],
                        y=[0 for m in sideways_mask if m],
                        name='Sideways',
                        mode='lines',
                        line=dict(color='gray', width=2),
                        connectgaps=False
                    ),
                    row=2, col=1
                )
            
            # Add regime bands
            fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=-0.5, line_dash="dash", line_color="red", row=2, col=1)
            
            # 3. RSI
            fig.add_trace(
                go.Scatter(
                    x=analysis_df['timestamp'],
                    y=[d['momentum_indicators']['rsi'] for d in analyzed_data],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # 4. MACD
            macd_data = [d['trend_indicators']['macd'] for d in analyzed_data]
            fig.add_trace(
                go.Scatter(
                    x=analysis_df['timestamp'],
                    y=[d['line'] for d in macd_data],
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=analysis_df['timestamp'],
                    y=[d['signal'] for d in macd_data],
                    name='Signal',
                    line=dict(color='orange', width=1)
                ),
                row=4, col=1
            )
            
            # MACD Histogram with colors
            histogram_values = [d['histogram'] for d in macd_data]
            histogram_colors = ['green' if v >= 0 else 'red' for v in histogram_values]
            
            fig.add_trace(
                go.Bar(
                    x=analysis_df['timestamp'],
                    y=histogram_values,
                    name='Histogram',
                    marker_color=histogram_colors
                ),
                row=4, col=1
            )
            
            # 5. Equity Curve
            fig.add_trace(
                go.Scatter(
                    x=analysis_df['timestamp'],
                    y=backtest_stats.equity_curve[1:],
                    name='Equity',
                    line=dict(color='blue', width=1)
                ),
                row=5, col=1
            )
            
            # Add performance metrics
            TradingSystemVisualizer._add_performance_metrics(fig, backtest_stats)
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text="Trading System Dashboard",
                showlegend=True,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            # Update axes labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Regime", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            fig.update_yaxes(title_text="Equity ($)", row=5, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            raise

    @staticmethod
    def _add_trade_markers(fig: go.Figure, positions: List['Position']) -> None:
        """Add trade entry and exit markers to the chart"""
        try:
            # Separate long and short positions
            long_entries = [p for p in positions if p.direction == 1]
            short_entries = [p for p in positions if p.direction == -1]
            
            # Plot long entries and exits
            if long_entries:
                fig.add_trace(
                    go.Scatter(
                        x=[p.entry_time for p in long_entries],
                        y=[p.entry_price for p in long_entries],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='Long Entry'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[p.exit_time for p in long_entries if p.exit_time],
                        y=[p.exit_price for p in long_entries if p.exit_price],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='Long Exit'
                    ),
                    row=1, col=1
                )
            
            # Plot short entries and exits
            if short_entries:
                fig.add_trace(
                    go.Scatter(
                        x=[p.entry_time for p in short_entries],
                        y=[p.entry_price for p in short_entries],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='Short Entry'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[p.exit_time for p in short_entries if p.exit_time],
                        y=[p.exit_price for p in short_entries if p.exit_price],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='Short Exit'
                    ),
                    row=1, col=1
                )
        except Exception as e:
            print(f"Warning: Failed to add trade markers: {str(e)}")

    @staticmethod
    def _add_performance_metrics(fig: go.Figure, backtest_stats: 'BacktestStats') -> None:
        """Add performance metrics annotation to the chart"""
        try:
            stats_text = (
                f"Total Return: {backtest_stats.total_return:.2%}<br>"
                f"Sharpe Ratio: {backtest_stats.sharpe_ratio:.2f}<br>"
                f"Max Drawdown: {backtest_stats.max_drawdown:.2%}<br>"
                f"Win Rate: {backtest_stats.win_rate:.2%}<br>"
                f"Total Trades: {backtest_stats.total_trades}<br>"
                f"Profitable Trades: {backtest_stats.profitable_trades}<br>"
                f"Avg Win: ${backtest_stats.avg_win:,.2f}<br>"
                f"Avg Loss: ${backtest_stats.avg_loss:,.2f}"
            )
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.02,
                y=0.98,
                text=stats_text,
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        except Exception as e:
            print(f"Warning: Failed to add performance metrics: {str(e)}")

    @staticmethod
    def save_dashboard(fig: go.Figure, filename: str = "trading_dashboard.html") -> None:
        """Save the dashboard to an HTML file with error handling"""
        try:
            fig.write_html(filename)
            print(f"Dashboard saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving dashboard: {str(e)}")
            raise