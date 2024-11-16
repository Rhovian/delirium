from datetime import datetime, timezone, timedelta
from typing import List, Dict
from supabase import create_client, Client
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

class CryptoDataCollector:
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the collector with Supabase credentials
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase API key
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.alpaca_client = CryptoHistoricalDataClient()  # No auth needed for crypto data
        
    def fetch_ohlc_data(self, symbol: str, days: int = 1) -> List[Dict]:
        """
        Fetch OHLC data from Alpaca
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            days: Number of days of historical data to fetch
        """
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            
            bars = self.alpaca_client.get_crypto_bars(request)
            
            data = []
            for bar in bars[symbol]:
                data.append({
                    "timestamp": bar.timestamp.isoformat(),
                    "symbol": symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume)
                })
            
            print(f"Fetched {len(data)} bars from Alpaca")
            return data
            
        except Exception as e:
            print(f"Error fetching data from Alpaca: {str(e)}")
            return []

    def store_data(self, data: List[Dict], table_name: str = 'crypto_ohlc') -> None:
        """
        Store the transformed data in TimescaleDB via Supabase
        """
        if not data:
            print("No data to store")
            return

        try:
            result = self.supabase.table(table_name).upsert(
                data,
                on_conflict='timestamp,symbol',
                returning='minimal'
            ).execute()
            
            print(f"Successfully stored {len(data)} records")
            
        except Exception as e:
            print(f"Error storing data: {str(e)}")
            raise

    def collect_and_store(self, symbol: str, days: int = 1) -> None:
        """
        Main method to orchestrate the collection and storage process
        """
        print(f"Starting collection for {symbol}...")
        
        data = self.fetch_ohlc_data(symbol, days)
        if not data:
            print("No data retrieved")
            return

        print(f"Retrieved {len(data)} records")
        self.store_data(data)

    def query_recent_data(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Query recent data for a given symbol
        """
        try:
            result = self.supabase.table('crypto_ohlc')\
                .select('*')\
                .eq('symbol', symbol)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
        except Exception as e:
            print(f"Error querying data: {e}")
            return []