import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, Activity, DollarSign, Zap, Eye, Shield, Smartphone, Wifi, RefreshCw, Target, BarChart3 } from 'lucide-react';

const CryptoTradingDashboard = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');
  const [selectedAsset, setSelectedAsset] = useState('BTC');
  const [livePrices, setLivePrices] = useState({});
  const [priceHistory, setPriceHistory] = useState({});

  // Simulate real-time updates
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Simulate live price updates
  useEffect(() => {
    const updatePrices = () => {
      const coins = ['BTC', 'ETH', 'SOL', 'HYPE', 'PEPE', 'ADA', 'LINK', 'MATIC'];
      const basePrices = {
        BTC: 44180, 
        ETH: 2445, 
        SOL: 101.50, 
        HYPE: 0.312, 
        PEPE: 0.000008, 
        ADA: 0.485, 
        LINK: 14.25, 
        MATIC: 0.892
      };

      setLivePrices(prev => {
        const newPrices = {};
        coins.forEach(coin => {
          const basePrice = basePrices[coin];
          const volatility = coin === 'PEPE' || coin === 'HYPE' ? 0.05 : 0.02;
          const change = (Math.random() - 0.5) * volatility;
          const currentPrice = prev[coin]?.price || basePrice;
          const newPrice = currentPrice * (1 + change);
          
          // Add funding rate for perp trading (Hyperliquid data)
          const fundingRate = (Math.random() - 0.5) * 0.01; // -0.5% to +0.5%
          
          newPrices[coin] = {
            price: newPrice,
            change24h: ((newPrice - basePrice) / basePrice) * 100,
            volume24h: Math.random() * 1000000000 + 500000000,
            marketCap: newPrice * (coin === 'BTC' ? 19500000 : coin === 'ETH' ? 120000000 : 1000000000),
            fundingRate: fundingRate,
            openInterest: Math.random() * 2000000000,
            source: coin === 'BTC' || coin === 'ETH' || coin === 'SOL' ? 'hyperliquid' : 'binance',
            lastUpdate: Date.now()
          };
        });
        return newPrices;
      });

      // Update price history for charts
      setPriceHistory(prev => {
        const newHistory = { ...prev };
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false }).slice(0, -3);
        
        coins.forEach(coin => {
          if (!newHistory[coin]) newHistory[coin] = [];
          newHistory[coin].push({
            time: timeStr,
            price: livePrices[coin]?.price || basePrices[coin],
            volume: Math.random() * 100000000,
            fundingRate: livePrices[coin]?.fundingRate || 0
          });
          // Keep only last 20 data points
          if (newHistory[coin].length > 20) {
            newHistory[coin] = newHistory[coin].slice(-20);
          }
        });
        return newHistory;
      });
    };

    // Initial price load
    updatePrices();
    
    // Update prices every 2 seconds for live feel
    const priceTimer = setInterval(updatePrices, 2000);
    return () => clearInterval(priceTimer);
  }, []);

  // Mock data - replace with real WebSocket data
  const portfolioData = [
    { time: '00:00', value: 10000, pnl: 0 },
    { time: '04:00', value: 10250, pnl: 250 },
    { time: '08:00', value: 10180, pnl: 180 },
    { time: '12:00', value: 10420, pnl: 420 },
    { time: '16:00', value: 10380, pnl: 380 },
    { time: '20:00', value: 10650, pnl: 650 },
    { time: '24:00', value: 10720, pnl: 720 }
  ];

  const sentimentData = [
    { time: '00:00', sentiment: 0.2, confidence: 0.8, volume: 1200 },
    { time: '04:00', sentiment: 0.4, confidence: 0.85, volume: 1400 },
    { time: '08:00', sentiment: 0.1, confidence: 0.7, volume: 1100 },
    { time: '12:00', sentiment: 0.6, confidence: 0.9, volume: 1800 },
    { time: '16:00', sentiment: 0.3, confidence: 0.75, volume: 1300 },
    { time: '20:00', sentiment: 0.7, confidence: 0.95, volume: 2100 },
    { time: '24:00', sentiment: 0.8, confidence: 0.92, volume: 2300 }
  ];

  const positions = [
    { asset: 'BTC', size: 0.25, entry: 43250, current: 44180, pnl: 232.50, pnlPercent: 2.15 },
    { asset: 'SOL', size: 45, entry: 98.20, current: 101.50, pnl: 148.50, pnlPercent: 3.36 },
    { asset: 'ETH', size: 2.1, entry: 2420, current: 2445, pnl: 52.50, pnlPercent: 1.03 },
    { asset: 'HYPE', size: 1000, entry: 0.285, current: 0.312, pnl: 27.00, pnlPercent: 9.47 }
  ];

  const alerts = [
    { id: 1, type: 'price', message: `🚀 ${selectedAsset} up ${livePrices[selectedAsset]?.change24h?.toFixed(2) || '0.00'}% in 24h`, time: 'Live', severity: 'high', source: 'hyperliquid' },
    { id: 2, type: 'funding', message: 'BTC funding rate extreme: -0.08% (shorts paying longs)', time: '3 min ago', severity: 'high', source: 'hyperliquid' },
    { id: 3, type: 'sentiment', message: 'Alpha Vantage: 247 bullish articles detected (64% positive)', time: '5 min ago', severity: 'medium', source: 'alpha_vantage' },
    { id: 4, type: 'signal', message: 'Strong buy signal: SOL spot-perp premium divergence', time: '8 min ago', severity: 'high', source: 'binance' },
    { id: 5, type: 'correlation', message: 'SPY +0.8% suggests risk-on sentiment for crypto', time: '12 min ago', severity: 'medium', source: 'alpha_vantage' },
    { id: 6, type: 'system', message: 'All APIs operational - latency <50ms', time: '15 min ago', severity: 'low', source: 'system' }
  ];

  const modelPerformance = [
    { name: 'CryptoBERT', accuracy: 89.2, confidence: 0.94 },
    { name: 'FinBERT', accuracy: 87.5, confidence: 0.91 },
    { name: 'RoBERTa', accuracy: 85.8, confidence: 0.88 },
    { name: 'Ensemble', accuracy: 91.3, confidence: 0.96 }
  ];

  const assetAllocation = [
    { name: 'BTC', value: 4500, color: '#F7931A' },
    { name: 'ETH', value: 2800, color: '#627EEA' },
    { name: 'SOL', value: 2200, color: '#9945FF' },
    { name: 'Meme Coins', value: 1200, color: '#FF6B6B' }
  ];

  const StatCard = ({ title, value, change, icon: Icon, color = "blue" }) => (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm">{title}</p>
          <p className="text-white text-xl font-bold">{value}</p>
          {change && (
            <div className={`flex items-center text-sm ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {change >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              <span className="ml-1">{Math.abs(change).toFixed(2)}%</span>
            </div>
          )}
        </div>
        <Icon className={`text-${color}-400`} size={24} />
      </div>
    </div>
  );

  const AlertItem = ({ alert }) => {
    const severityColors = {
      high: 'border-red-500 bg-red-900/20',
      medium: 'border-yellow-500 bg-yellow-900/20',
      low: 'border-blue-500 bg-blue-900/20'
    };

    const getSourceBadge = (source) => {
      const badges = {
        'hyperliquid': { color: 'bg-purple-600', text: 'HL' },
        'binance': { color: 'bg-yellow-600', text: 'BN' },
        'alpha_vantage': { color: 'bg-blue-600', text: 'AV' },
        'system': { color: 'bg-gray-600', text: 'SYS' }
      };
      return badges[source] || { color: 'bg-gray-600', text: 'API' };
    };

    const sourceBadge = getSourceBadge(alert.source);

    return (
      <div className={`border-l-4 p-3 mb-2 rounded ${severityColors[alert.severity]}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center flex-1">
            <p className="text-white text-sm flex-1">{alert.message}</p>
            <span className={`ml-2 px-2 py-1 rounded text-xs font-bold ${sourceBadge.color}`}>
              {sourceBadge.text}
            </span>
          </div>
          <span className="text-gray-400 text-xs ml-2">{alert.time}</span>
        </div>
      </div>
    );
  };

  const LiveCoinTicker = ({ coin, data }) => {
    const isPositive = data?.change24h >= 0;
    const isFundingPositive = data?.fundingRate >= 0;
    const formatPrice = (price) => {
      if (price < 0.001) return price?.toFixed(8);
      if (price < 1) return price?.toFixed(6);
      if (price < 100) return price?.toFixed(4);
      return price?.toFixed(2);
    };

    const getSourceBadge = (source) => {
      const badges = {
        'hyperliquid': { color: 'bg-purple-600', text: 'HL' },
        'binance': { color: 'bg-yellow-600', text: 'BN' },
        'dexscreener': { color: 'bg-green-600', text: 'DEX' }
      };
      return badges[source] || { color: 'bg-gray-600', text: 'API' };
    };

    const sourceBadge = getSourceBadge(data?.source);

    return (
      <div className="bg-gray-800 rounded-lg p-3 min-w-[220px] border border-gray-700 hover:border-gray-600 transition-colors">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-2 ${isPositive ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
            <h3 className="text-white font-bold text-lg">{coin}</h3>
            <span className={`ml-2 px-2 py-1 rounded text-xs font-bold ${sourceBadge.color}`}>
              {sourceBadge.text}
            </span>
          </div>
          {isPositive ? <TrendingUp className="text-green-400" size={16} /> : <TrendingDown className="text-red-400" size={16} />}
        </div>
        
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Price</span>
            <span className="text-white font-mono text-sm">${formatPrice(data?.price)}</span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">24h</span>
            <span className={`font-mono text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {isPositive ? '+' : ''}{data?.change24h?.toFixed(2)}%
            </span>
          </div>
          
          {data?.fundingRate !== undefined && (
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Funding</span>
              <span className={`font-mono text-xs ${isFundingPositive ? 'text-red-400' : 'text-green-400'}`}>
                {isFundingPositive ? '+' : ''}{(data.fundingRate * 100).toFixed(3)}%
              </span>
            </div>
          )}
          
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Vol</span>
            <span className="text-gray-300 text-xs">
              ${data?.volume24h ? (data.volume24h / 1000000).toFixed(1) : '0'}M
            </span>
          </div>
        </div>
        
        <div className="mt-3 pt-2 border-t border-gray-700">
          <div className="flex space-x-2">
            <button 
              onClick={() => setSelectedAsset(coin)}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-xs py-1 px-2 rounded transition-colors"
            >
              Chart
            </button>
            <button className="flex-1 bg-green-600 hover:bg-green-700 text-xs py-1 px-2 rounded transition-colors">
              Long
            </button>
            <button className="flex-1 bg-red-600 hover:bg-red-700 text-xs py-1 px-2 rounded transition-colors">
              Short
            </button>
          </div>
        </div>
      </div>
    );
  };

  const CoinChart = ({ coin, data }) => {
    if (!data || data.length === 0) return (
      <div className="flex items-center justify-center h-32 text-gray-400">
        Loading {coin} data...
      </div>
    );

    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-white font-bold">{coin} Price Chart</h3>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-green-400 text-sm">Live</span>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" tickFormatter={(value) => `${value.toFixed(value < 1 ? 6 : 2)}`} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
              labelStyle={{ color: '#F3F4F6' }}
              formatter={(value) => [`${value.toFixed(value < 1 ? 6 : 2)}`, 'Price']}
            />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#10B981" 
              strokeWidth={2} 
              dot={false}
              strokeDasharray={selectedAsset === coin ? "0" : "5 5"}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black text-white p-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-white">Trading Command Center</h1>
          <p className="text-gray-400">{currentTime.toLocaleString()}</p>
        </div>
        <div className="flex items-center space-x-4 mt-2 md:mt-0">
          <div className="flex items-center space-x-2">
            <div className="flex items-center text-purple-400">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse mr-1" />
              <span className="text-xs font-medium">Hyperliquid</span>
            </div>
            <div className="flex items-center text-yellow-400">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-1" />
              <span className="text-xs font-medium">Binance</span>
            </div>
            <div className="flex items-center text-blue-400">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-1" />
              <span className="text-xs font-medium">Alpha Vantage</span>
            </div>
          </div>
          <div className="bg-gray-800 px-3 py-1 rounded text-sm">
            GPU: 87% | Models: Active
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          title="Portfolio Value"
          value="$10,720"
          change={7.2}
          icon={DollarSign}
          color="green"
        />
        <StatCard
          title="24h P&L"
          value="$720"
          change={6.8}
          icon={TrendingUp}
          color="green"
        />
        <StatCard
          title="Active Positions"
          value="4"
          icon={Activity}
          color="blue"
        />
        <StatCard
          title="Win Rate"
          value="89.2%"
          change={2.1}
          icon={Zap}
          color="yellow"
        />
      </div>

      {/* Live Coin Tickers */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Live Market Data</h2>
          <div className="flex items-center text-green-400">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2" />
            <span className="text-sm">Real-time</span>
          </div>
        </div>
        
        <div className="overflow-x-auto pb-4">
          <div className="flex space-x-4 min-w-max">
            {['BTC', 'ETH', 'SOL', 'HYPE', 'PEPE', 'ADA', 'LINK', 'MATIC'].map(coin => (
              <LiveCoinTicker 
                key={coin} 
                coin={coin} 
                data={livePrices[coin]}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Selected Asset Chart */}
      {priceHistory[selectedAsset] && (
        <div className="mb-6">
          <CoinChart coin={selectedAsset} data={priceHistory[selectedAsset]} />
        </div>
      )}

      {/* Asset Allocation & Portfolio Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Asset Allocation */}
        <div className="bg-gray-900 rounded-lg p-4">
          <h2 className="text-xl font-bold mb-4">Portfolio Allocation</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={assetAllocation}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {assetAllocation.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`${value}`, 'Value']} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Traditional Market Correlations */}
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">📈 Market Correlations</h2>
            <div className="flex items-center text-blue-400">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-2" />
              <span className="text-sm">Alpha Vantage</span>
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
              <div className="flex items-center">
                <span className="text-white font-medium">SPY</span>
                <span className="text-gray-400 text-sm ml-2">S&P 500</span>
              </div>
              <div className="text-right">
                <p className="text-white font-mono">$428.54</p>
                <p className="text-green-400 text-sm">+0.8%</p>
              </div>
            </div>
            
            <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
              <div className="flex items-center">
                <span className="text-white font-medium">QQQ</span>
                <span className="text-gray-400 text-sm ml-2">Nasdaq</span>
              </div>
              <div className="text-right">
                <p className="text-white font-mono">$385.12</p>
                <p className="text-green-400 text-sm">+1.2%</p>
              </div>
            </div>
            
            <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
              <div className="flex items-center">
                <span className="text-white font-medium">DXY</span>
                <span className="text-gray-400 text-sm ml-2">US Dollar</span>
              </div>
              <div className="text-right">
                <p className="text-white font-mono">103.45</p>
                <p className="text-red-400 text-sm">-0.3%</p>
              </div>
            </div>
            
            <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
              <div className="flex items-center">
                <span className="text-white font-medium">VIX</span>
                <span className="text-gray-400 text-sm ml-2">Fear Index</span>
              </div>
              <div className="text-right">
                <p className="text-white font-mono">16.2</p>
                <p className="text-red-400 text-sm">-2.1%</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Performance & Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Model Performance */}
        <div className="bg-gray-900 rounded-lg p-4">
          <h2 className="text-xl font-bold mb-4">AI Model Performance</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={modelPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="accuracy" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Alerts */}
        <div className="bg-gray-900 rounded-lg p-4">
          <h2 className="text-xl font-bold mb-4">Real-time Alerts</h2>
          <div className="max-h-48 overflow-y-auto">
            {alerts.map(alert => (
              <AlertItem key={alert.id} alert={alert} />
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-900 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="bg-red-600 hover:bg-red-700 p-3 rounded-lg text-center transition-colors">
            <Shield className="mx-auto mb-2" size={20} />
            <span className="text-sm">Emergency Stop</span>
          </button>
          <button className="bg-blue-600 hover:bg-blue-700 p-3 rounded-lg text-center transition-colors">
            <RefreshCw className="mx-auto mb-2" size={20} />
            <span className="text-sm">Rebalance</span>
          </button>
          <button className="bg-purple-600 hover:bg-purple-700 p-3 rounded-lg text-center transition-colors">
            <Eye className="mx-auto mb-2" size={20} />
            <span className="text-sm">Watch Mode</span>
          </button>
          <button className="bg-green-600 hover:bg-green-700 p-3 rounded-lg text-center transition-colors">
            <Target className="mx-auto mb-2" size={20} />
            <span className="text-sm">Set Alerts</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default CryptoTradingDashboard; 