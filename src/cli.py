"""
cli.py - Enhanced CLI with backtesting capabilities
"""
import os
import click
from dotenv import load_dotenv
from src.collector import CryptoDataCollector
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from src.collector import CryptoDataCollector
from src.analyze import QuantAnalyzer, SignalGenerator, BacktestEngine
from src.viz import TradingSystemVisualizer

def init_client():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not all([SUPABASE_URL, SUPABASE_KEY]):
        raise ValueError("Missing required environment variables")
    
    return SUPABASE_URL, SUPABASE_KEY

@click.group()
def cli():
    """Enhanced crypto trading CLI with backtesting"""
    pass

# Keep your existing commands
@cli.command()
@click.option('--symbol', default="BTC/USD", help='Trading pair symbol')
@click.option('--days', default=30, help='Days of historical data to fetch')
def ingest(symbol: str, days: int):
    """Ingest historical price data"""
    try:
        url, key = init_client()
        collector = CryptoDataCollector(url, key)
        
        # Check latest stored record
        latest = collector.query_recent_data(symbol, limit=1)
        if latest:
            latest_timestamp = datetime.fromisoformat(latest[0]['timestamp'])
            print(f"Latest stored data is from: {latest_timestamp}")
            
            # Only fetch newer data
            now = datetime.now(timezone.utc)
            days_needed = (now - latest_timestamp).days + 1
            if days_needed < 1:
                print("Data is up to date")
                return
            days = min(days, days_needed)
        
        print(f"Fetching {days} days of data for {symbol}...")
        collector.collect_and_store(symbol, days=days)
        
        # Verify
        recent = collector.query_recent_data(symbol, limit=5)
        print("\nMost recent records:")
        for record in recent:
            print(f"{record['timestamp']}: ${record['close']:,.2f}")
            
    except Exception as e:
        print(f"Error: {e}")

@cli.command()
@click.option('--symbol', default="BTC/USD", help='Trading pair symbol')
@click.option('--days', default=365, help='Days of historical data to analyze')
@click.option('--initial-capital', default=100000, help='Initial capital for backtesting')
@click.option('--risk-per-trade', default=0.02, type=float, help='Risk per trade as decimal')
@click.option('--output-dir', default='./backtests', help='Directory for output files')
def backtest(symbol: str, days: int, initial_capital: float, risk_per_trade: float, output_dir: str):
    """Run backtesting analysis on historical data"""
    try:
        # Initialize components
        url, key = init_client()
        collector = CryptoDataCollector(url, key)
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Fetch historical data
        print(f"Fetching {days} days of historical data for {symbol}...")
        data = collector.query_recent_data(symbol, limit=days * 24)  # Assuming hourly data
        
        if not data:
            print("No data available for analysis")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"\nAnalyzing {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Initialize trading system components
        analyzer = QuantAnalyzer()
        signal_gen = SignalGenerator(weights={
            'trend': 0.4,
            'momentum': 0.2,
            'volatility': 0.2,
            'regime': 0.2
        })
        backtest_engine = BacktestEngine(analyzer, signal_gen)
        
        # Run backtest
        print("\nRunning backtest...")
        backtest_stats = backtest_engine.run(
            df,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade
        )
        
        # Create visualization
        print("\nGenerating visualization...")
        analyzed_data = analyzer.analyze_data(df)
        fig = TradingSystemVisualizer.create_dashboard(
            df,
            analyzed_data,
            backtest_stats
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = os.path.join(output_dir, f'backtest_{symbol.replace("/", "_")}_{timestamp}.html')
        TradingSystemVisualizer.save_dashboard(fig, viz_path)
        
        # Print summary
        print("\nBacktest Results:")
        print(f"Total Return: {backtest_stats.total_return:.2%}")
        print(f"Sharpe Ratio: {backtest_stats.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {backtest_stats.max_drawdown:.2%}")
        print(f"Win Rate: {backtest_stats.win_rate:.2%}")
        print(f"Total Trades: {backtest_stats.total_trades}")
        print(f"Profitable Trades: {backtest_stats.profitable_trades}")
        print(f"Average Win: ${backtest_stats.avg_win:,.2f}")
        print(f"Average Loss: ${backtest_stats.avg_loss:,.2f}")
        
        print(f"\nVisualization saved to: {viz_path}")
        
        # Optional: Save detailed trade history
        trades_df = pd.DataFrame([
            {
                'entry_time': p.entry_time,
                'exit_time': p.exit_time,
                'entry_price': p.entry_price,
                'exit_price': p.exit_price,
                'size': p.size,
                'direction': p.direction,
                'pnl': p.pnl
            }
            for p in backtest_stats.positions
        ])
        
        trades_path = os.path.join(output_dir, f'trades_{symbol.replace("/", "_")}_{timestamp}.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"Trade history saved to: {trades_path}")
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        raise

if __name__ == '__main__':
    cli()