import asyncio
from datetime import datetime
from api.exchanges.hyperliquid import HyperliquidClient
from api.exchanges.binance import BinanceClient
from api.exchanges.okx import OKXClient
from exchanges.exchange_config import get_exchange_config

class OrderExecutor:
    def __init__(self):
        # Load all exchange credentials securely
        config = get_exchange_config()
        hl_cfg = config.get_hyperliquid_config()
        binance_cfg = config.get_binance_config()
        okx_cfg = config.get_okx_config()

        if hl_cfg and hl_cfg.wallet_address and hl_cfg.private_key:
            self.hyperliquid = HyperliquidClient(api_key=hl_cfg.private_key, wallet_address=hl_cfg.wallet_address)
        else:
            print("[WARNING] Hyperliquid credentials not found in secure config. Hyperliquid trading disabled.")
            self.hyperliquid = HyperliquidClient()

        if binance_cfg and binance_cfg.api_key and binance_cfg.secret_key:
            self.binance = BinanceClient(api_key=binance_cfg.api_key, api_secret=binance_cfg.secret_key)
        else:
            print("[WARNING] Binance credentials not found in secure config. Binance trading disabled.")
            self.binance = BinanceClient()

        if okx_cfg and okx_cfg.api_key and okx_cfg.secret_key:
            self.okx = OKXClient(api_key=okx_cfg.api_key, api_secret=okx_cfg.secret_key, passphrase=okx_cfg.passphrase)
        else:
            print("[WARNING] OKX credentials not found in secure config. OKX trading disabled.")
            self.okx = OKXClient()

        self.active_orders = {}
        
    async def initialize(self):
        """Connect to all exchanges"""
        await self.hyperliquid.connect()
        await self.binance.connect()
        await self.okx.connect()
        
    async def execute_trade(self, signal, position_size, symbol):
        """
        Smart order routing to best exchange
        """
        try:
            # Determine best exchange for the trade
            best_exchange = await self._select_best_exchange(symbol, position_size)
            
            # Execute the trade
            if symbol == "HYPE" and best_exchange == "hyperliquid":
                result = await self._execute_hyperliquid_trade(signal, position_size, symbol)
            elif symbol in ["SOL/USDT", "BTC/USDT", "ETH/USDT"]:
                if best_exchange == "binance":
                    result = await self._execute_binance_trade(signal, position_size, symbol)
                else:
                    result = await self._execute_okx_trade(signal, position_size, symbol)
            else:
                print(f"Unsupported symbol: {symbol}")
                return None
            
            # Track the order
            if result:
                self.active_orders[result['order_id']] = {
                    'symbol': symbol,
                    'exchange': best_exchange,
                    'signal': signal,
                    'timestamp': datetime.now(),
                    'status': 'pending'
                }
            
            return result
            
        except Exception as e:
            print(f"Order execution error: {e}")
            return None
    
    async def _select_best_exchange(self, symbol, size):
        """Choose best exchange based on liquidity and fees"""
        if symbol == "HYPE":
            return "hyperliquid"  # Only available on Hyperliquid
        
        # For other symbols, check liquidity and fees
        exchanges_data = {}
        
        try:
            # Get market data from each exchange
            binance_data = await self.binance.get_sol_data() if symbol == "SOL/USDT" else None
            okx_data = await self.okx.get_market_data(symbol)
            
            if binance_data:
                exchanges_data['binance'] = {
                    'spread': abs(binance_data['ask'] - binance_data['bid']) if binance_data['ask'] and binance_data['bid'] else float('inf'),
                    'volume': binance_data['volume']
                }
            
            if okx_data:
                exchanges_data['okx'] = {
                    'spread': 0.001,  # Placeholder
                    'volume': okx_data['volume']
                }
            
            # Choose exchange with best spread and sufficient volume
            best_exchange = min(exchanges_data.keys(), 
                              key=lambda x: exchanges_data[x]['spread']) if exchanges_data else 'binance'
            
            return best_exchange
            
        except Exception as e:
            print(f"Exchange selection error: {e}")
            return 'binance'  # Default fallback
    
    async def _execute_hyperliquid_trade(self, signal, size, symbol):
        """Execute trade on Hyperliquid"""
        try:
            side = 'buy' if signal['signal'] == 'buy' else 'sell'
            
            result = await self.hyperliquid.place_order(
                symbol=symbol,
                side=side,
                amount=size
            )
            
            return {
                'order_id': f"HL_{datetime.now().timestamp()}",
                'exchange': 'hyperliquid',
                'symbol': symbol,
                'side': side,
                'amount': size,
                'status': 'submitted',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Hyperliquid execution error: {e}")
            return None
    
    async def _execute_binance_trade(self, signal, size, symbol):
        """Execute trade on Binance"""
        try:
            side = 'buy' if signal['signal'] == 'buy' else 'sell'
            
            # Calculate amount in base currency (SOL for SOL/USDT)
            market_data = await self.binance.get_sol_data()
            if not market_data:
                return None
                
            current_price = market_data['price']
            amount = size / current_price  # Convert USDT size to SOL amount
            
            result = await self.binance.place_sol_order(side, amount)
            
            if result:
                return {
                    'order_id': result.get('id', f"BN_{datetime.now().timestamp()}"),
                    'exchange': 'binance',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': current_price,
                    'status': result.get('status', 'submitted'),
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            print(f"Binance execution error: {e}")
            return None
    
    async def _execute_okx_trade(self, signal, size, symbol):
        """Execute trade on OKX"""
        try:
            side = 'buy' if signal['signal'] == 'buy' else 'sell'
            
            # Get current price for amount calculation
            market_data = await self.okx.get_market_data(symbol)
            if not market_data:
                return None
                
            current_price = market_data['price']
            amount = size / current_price
            
            # Note: Implement actual OKX order placement
            return {
                'order_id': f"OKX_{datetime.now().timestamp()}",
                'exchange': 'okx',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': current_price,
                'status': 'submitted',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"OKX execution error: {e}")
            return None
    
    async def check_order_status(self, order_id):
        """Check status of active orders"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            # Implement actual status checking with exchange APIs
            return order
        return None
    
    async def cancel_order(self, order_id):
        """Cancel pending order"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            # Implement actual order cancellation
            del self.active_orders[order_id]
            return True
        return False 