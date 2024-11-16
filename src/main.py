import os
import click
from dotenv import load_dotenv
from src.collector import CryptoDataCollector
from src.analyzer import PriceAnalyzer
from supabase import create_client
from datetime import datetime, timezone

def init_client():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not all([SUPABASE_URL, SUPABASE_KEY]):
        raise ValueError("Missing required environment variables")
    
    return SUPABASE_URL, SUPABASE_KEY

@click.group()
def cli():
    """Crypto data management CLI"""
    pass

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
@click.option('--force', is_flag=True, help='Force reanalysis of all data')
@click.option('--batch-size', default=100, help='Number of records to process at once')
def analyze(symbol: str, force: bool, batch_size: int):
    """Calculate technical indicators for stored data"""
    try:
        url, key = init_client()
        collector = CryptoDataCollector(url, key)
        
        # Initial status
        initial_count = collector.supabase.table('crypto_ohlc')\
            .select('count', count='exact')\
            .eq('symbol', symbol)\
            .execute()
            
        initial_analyzed = collector.supabase.table('crypto_ohlc')\
            .select('count', count='exact')\
            .eq('symbol', symbol)\
            .not_.is_('trend_indicators', None)\
            .execute()
            
        total_records = initial_count.count
        analyzed_records = initial_analyzed.count
        needs_analysis = total_records - analyzed_records
        
        print(f"\nInitial Status:")
        print(f"Total records: {total_records}")
        print(f"Already analyzed: {analyzed_records}")
        print(f"Needs analysis: {needs_analysis}")
        
        if needs_analysis == 0 and not force:
            print("No records need analysis")
            return
            
        if force:
            print("\nForce flag set - will reanalyze all records")
            total_to_process = total_records
        else:
            total_to_process = needs_analysis
            
        print(f"\nWill process {total_to_process} records")
        
        # Keep track of successfully processed records
        successfully_processed = 0
        attempted = 0
        
        while True:
            # Get unanalyzed records count
            current_unanalyzed = collector.supabase.table('crypto_ohlc')\
                .select('count', count='exact')\
                .eq('symbol', symbol)\
                .is_('trend_indicators', None)\
                .execute()
                
            if current_unanalyzed.count == 0:
                print("\nAll records have been analyzed")
                break
                
            # Get next batch of timestamps
            query = collector.supabase.table('crypto_ohlc')\
                .select('timestamp')\
                .eq('symbol', symbol)\
                .is_('trend_indicators', None)\
                .order('timestamp')\
                .limit(batch_size)
                
            batch = query.execute()
            
            if not batch.data:
                print(f"\nNo more unanalyzed records found")
                break
                
            timestamps = [row['timestamp'] for row in batch.data]
            first_timestamp = min(timestamps)
            
            # Get historical context
            lookback = PriceAnalyzer.LOOKBACK_REQUIREMENTS['sma_200']
            context = collector.supabase.table('crypto_ohlc')\
                .select('*')\
                .eq('symbol', symbol)\
                .lt('timestamp', first_timestamp)\
                .order('timestamp', desc=True)\
                .limit(lookback)\
                .execute()
            
            # Get batch records in smaller chunks
            batch_records = []
            chunk_size = 20
            for i in range(0, len(timestamps), chunk_size):
                timestamp_chunk = timestamps[i:i + chunk_size]
                chunk_query = collector.supabase.table('crypto_ohlc')\
                    .select('*')\
                    .eq('symbol', symbol)\
                    .in_('timestamp', timestamp_chunk)\
                    .execute()
                batch_records.extend(chunk_query.data)
            
            if context.data and batch_records:
                all_data = context.data + batch_records
                print(f"\nProcessing batch of {len(batch_records)} records "
                      f"(with {len(context.data)} historical records)")
                
                analyzed_data = PriceAnalyzer.analyze_for_algorithm(all_data)
                
                # Only store the records we wanted to process
                to_store = [
                    record for record in analyzed_data 
                    if record['timestamp'] in timestamps
                ]
                
                if to_store:
                    collector.store_data(to_store)
                    successfully_processed += len(to_store)
                    print(f"Updated {len(to_store)} records with indicators")
                
            attempted += len(batch.data)
            
            # Get current progress
            current_analyzed = collector.supabase.table('crypto_ohlc')\
                .select('count', count='exact')\
                .eq('symbol', symbol)\
                .not_.is_('trend_indicators', None)\
                .execute()
                
            progress = (current_analyzed.count - analyzed_records) / needs_analysis * 100
            print(f"Overall progress: {progress:.1f}% "
                  f"({successfully_processed}/{needs_analysis} records)")
        
        # Final status
        final_analyzed = collector.supabase.table('crypto_ohlc')\
            .select('count', count='exact')\
            .eq('symbol', symbol)\
            .not_.is_('trend_indicators', None)\
            .execute()
            
        print(f"\nFinal Status:")
        print(f"Total records: {total_records}")
        print(f"Analyzed records: {final_analyzed.count}")
        print(f"Successfully processed in this run: {successfully_processed}")
        
        # Verify a recent record
        recent = collector.query_recent_data(symbol, limit=1)
        if recent and recent[0]:
            latest = recent[0]
            print("\nLatest analyzed record:")
            print(f"Timestamp: {latest.get('timestamp')}")
            print(f"Price: ${float(latest.get('close', 0)):,.2f}")
            
            trend = latest.get('trend_indicators', {})
            momentum = latest.get('momentum_indicators', {})
            
            if trend and momentum:
                print("Indicators:", {
                    'rsi': momentum.get('rsi', 'N/A'),
                    'sma_200': trend.get('sma_200', 'N/A')
                })
                
    except Exception as e:
        print(f"Error: {e}")

@cli.command()
@click.option('--symbol', default="BTC/USD", help='Trading pair symbol')
def status(symbol: str):
    """Check data status"""
    try:
        url, key = init_client()
        collector = CryptoDataCollector(url, key)
        
        # Get total records
        total = collector.supabase.table('crypto_ohlc')\
            .select('count', count='exact')\
            .eq('symbol', symbol)\
            .execute()
            
        # Get analyzed records
        analyzed = collector.supabase.table('crypto_ohlc')\
            .select('count', count='exact')\
            .eq('symbol', symbol)\
            .not_.is_('trend_indicators', None)\
            .execute()
            
        # Get latest timestamp
        latest = collector.query_recent_data(symbol, limit=1)
        
        print(f"\nStatus for {symbol}:")
        print(f"Total records: {total.count}")
        print(f"Analyzed records: {analyzed.count}")
        print(f"Records needing analysis: {total.count - analyzed.count}")
        if latest:
            print(f"Latest data from: {latest[0]['timestamp']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    cli()