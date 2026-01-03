<x-app-layout>

    <script src="{{ asset('js/lightweight-charts.js?v=2') }}" defer onerror="console.error('Failed to load local chart lib'); document.getElementById('chart-debug').innerText = 'Failed to load JS file';"></script>
    <div class="h-[calc(100vh-5rem)] flex flex-col lg:flex-row gap-6 p-4 md:p-6" x-data="alphaTerminal()">
        
        <!-- Left Panel: Charting & Execution (Aurora Glass) -->
        <div class="flex-1 flex flex-col bg-[#0a0a0a]/80 backdrop-blur-xl border border-white/10 rounded-3xl overflow-hidden relative shadow-[0_0_50px_-12px_rgba(124,58,237,0.1)]">
            
            <!-- Top Bar: Asset Selector & Stats -->
            <div class="h-16 border-b border-white/5 flex items-center justify-between px-6 bg-white/5 relative z-20">
                <div class="flex items-center gap-6">
                    <!-- Asset Dropdown -->
                    <div class="relative" x-data="{ open: false }">
                        <button @click="open = !open" @click.away="open = false" class="flex items-center gap-3 group focus:outline-none">
                            <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
                                <svg class="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
                            </div>
                            <div class="text-left">
                                <h2 class="text-white font-bol text-lg leading-none flex items-center gap-2">
                                    <span x-text="activeAsset"></span>
                                    <svg class="w-3 h-3 text-gray-500 group-hover:text-violet-400 transition-colors" :class="open ? 'rotate-180' : ''" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
                                </h2>
                                <span class="text-[10px] text-gray-400 font-medium tracking-wider" x-text="assets[activeAsset].name"></span>
                            </div>
                        </button>

                        <!-- Dropdown Menu -->
                        <div x-show="open" 
                             x-transition:enter="transition ease-out duration-200"
                             x-transition:enter-start="opacity-0 translate-y-2"
                             x-transition:enter-end="opacity-100 translate-y-0"
                             x-transition:leave="transition ease-in duration-150"
                             class="absolute top-full left-0 mt-3 w-64 bg-[#111] border border-white/10 rounded-xl shadow-2xl py-2 z-50 backdrop-blur-xl">
                            <div class="px-4 py-2 text-xs font-bold text-gray-500 uppercase tracking-widest">Markets</div>
                            <template x-for="(details, symbol) in assets" :key="symbol">
                                <button @click="switchAsset(symbol); open = false" 
                                    class="w-full px-4 py-3 text-left hover:bg-white/5 transition-colors flex items-center justify-between group border-l-2 border-transparent hover:border-violet-500">
                                    <div class="flex items-center gap-3">
                                        <span class="text-sm font-bold text-gray-300 group-hover:text-white" x-text="symbol"></span>
                                    </div>
                                    <span class="text-xs text-gray-500 group-hover:text-violet-400" x-text="details.price"></span>
                                </button>
                            </template>
                        </div>
                    </div>

                    <!-- Live Ticker -->
                    <div class="hidden md:flex items-center gap-6 border-l border-white/10 pl-6">
                        <div>
                            <p class="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Market Price</p>
                            <p class="text-lg font-mono font-bold text-white tracking-tight" x-text="formatPrice(assets[activeAsset].price)"></p>
                        </div>
                        <div>
                            <p class="text-[10px] text-gray-500 font-bold uppercase tracking-wider">24h Change</p>
                            <p class="text-sm font-mono font-bold text-emerald-400 flex items-center gap-1">
                                <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
                                +1.24%
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Controls -->
                <div class="flex items-center gap-3">
                    <div class="flex bg-black/40 rounded-lg p-1 border border-white/5">
                        <template x-for="tf in ['1M', '15M', '1H', '4H', '1D']">
                            <button class="px-3 py-1.5 text-xs rounded-md font-medium transition-all"
                                :class="activeTimeframe === tf ? 'bg-violet-600 text-white shadow-lg shadow-violet-500/25' : 'text-gray-500 hover:text-gray-300'"
                                @click="setTimeframe(tf)" x-text="tf"></button>
                        </template>
                    </div>
                    <button @click="fullscreen = !fullscreen" class="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors">
                        <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>
                    </button>
                </div>
            </div>

            <!-- Chart Area -->
            <div class="flex-1 relative w-full bg-[#050505] overflow-hidden">
                <div id="alphaChart" class="absolute inset-0 w-full h-full"></div>
                
                <!-- Scanner Overlay (Premium HUD Style) -->
                <div x-show="isScanning" 
                     x-transition:enter="transition ease-out duration-300"
                     x-transition:enter-start="opacity-0 scale-95"
                     x-transition:enter-end="opacity-100 scale-100"
                     x-transition:leave="transition ease-in duration-200"
                     x-transition:leave-start="opacity-100 scale-100"
                     x-transition:leave-end="opacity-0 scale-95"
                     class="absolute inset-0 z-30 pointer-events-none flex flex-col justify-between p-6 bg-black/40 backdrop-blur-[2px]">
                     
                     <!-- Background Grid -->
                     <div class="absolute inset-0 opacity-20" style="background-image: linear-gradient(rgba(34, 211, 238, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(34, 211, 238, 0.1) 1px, transparent 1px); background-size: 40px 40px;"></div>

                     <!-- The Scan Beam (Moves Up/Down) -->
                     <div class="absolute inset-x-0 h-32 -translate-y-1/2 pointer-events-none animate-[scanPremium_2s_ease-in-out_infinite]">
                         <div class="w-full h-full bg-gradient-to-b from-transparent via-cyan-500/10 to-transparent"></div>
                         <div class="absolute top-1/2 left-0 right-0 h-[2px] bg-cyan-400 shadow-[0_0_40px_4px_rgba(34,211,238,0.5)]"></div>
                     </div>

                     <!-- HUD Corners -->
                     <div class="flex justify-between relative z-10">
                         <div class="w-16 h-16 border-t-2 border-l-2 border-cyan-500 rounded-tl-xl shadow-[0_0_15px_rgba(34,211,238,0.5)]"></div>
                         <div class="w-16 h-16 border-t-2 border-r-2 border-cyan-500 rounded-tr-xl shadow-[0_0_15px_rgba(34,211,238,0.5)]"></div>
                     </div>
                     
                     <!-- Center Status -->
                     <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center z-20">
                         <h2 class="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-white tracking-tighter drop-shadow-[0_0_10px_rgba(34,211,238,0.8)]">
                             AI SCANNING
                         </h2>
                         <div class="flex items-center justify-center gap-2 mt-4">
                             <span class="w-2 h-2 bg-cyan-400 rounded-full animate-ping"></span>
                             <p class="text-cyan-400 font-mono text-sm tracking-[0.3em] font-bold">ANALYZING PATTERNS</p>
                         </div>
                     </div>

                     <div class="flex justify-between relative z-10">
                         <div class="w-16 h-16 border-b-2 border-l-2 border-cyan-500 rounded-bl-xl shadow-[0_0_15px_rgba(34,211,238,0.5)]"></div>
                         <div class="w-16 h-16 border-b-2 border-r-2 border-cyan-500 rounded-br-xl shadow-[0_0_15px_rgba(34,211,238,0.5)]"></div>
                     </div>
                </div>
                
                <!-- Status Overlay (Loading/Error) -->
                <div x-show="status.loading || status.error" 
                     class="absolute inset-0 bg-[#050505]/80 backdrop-blur-sm flex items-center justify-center z-20">
                    <div class="text-center">
                        <template x-if="status.loading">
                             <div>
                                <svg class="animate-spin h-8 w-8 text-violet-500 mx-auto mb-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <p class="text-gray-400 text-sm font-mono animate-pulse">Initializing Alpha Chart...</p>
                                <p id="chart-debug" class="text-xs text-gray-600 mt-1" x-text="status.debug"></p>
                             </div>
                        </template>
                        <template x-if="status.error">
                            <div class="text-red-500">
                                <svg class="w-8 h-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                <p class="font-bold">Chart Error</p>
                                <p class="text-sm opacity-75" x-text="status.message"></p>
                            </div>
                        </template>
                    </div>
                </div>


            </div>
        </div>

        <!-- Right Panel: PipFlow AI Chat (Collapsible) -->
        <div class="w-full lg:w-[400px] flex flex-col bg-[#0a0a0a]/80 backdrop-blur-xl border border-white/10 rounded-3xl overflow-hidden shadow-[0_0_50px_-12px_rgba(6,182,212,0.1)] relative"
             x-show="!fullscreen"
             x-transition:enter="transition ease-out duration-300"
             x-transition:enter-start="opacity-0 translate-x-10"
             x-transition:enter-end="opacity-100 translate-x-0">
            
            <!-- AI Header -->
            <div class="h-16 border-b border-white/5 flex items-center justify-between px-6 bg-white/5">
                <div class="flex items-center gap-3">
                    <div class="relative">
                        <div class="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#22d3ee] animate-pulse"></div>
                        <div class="absolute inset-0 w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping opacity-75"></div>
                    </div>
                    <h3 class="text-white font-bold tracking-tight">PipFlow AI</h3>
                </div>
                <div class="px-2 py-1 rounded bg-white/5 border border-white/5 text-[10px] font-mono text-cyan-400">ONLINE</div>
            </div>

            <!-- Chat Stream -->
            <div class="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar" x-ref="chatStream">
                <!-- AI Greeting -->
                <div class="flex items-start gap-4 animate-fade-in-up">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-cyan-500/20">
                        <svg class="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                    </div>
                    <div class="space-y-1">
                        <span class="text-[10px] font-bold text-cyan-400 ml-1">PIPFLOW</span>
                        <div class="bg-white/5 border border-white/5 rounded-2xl rounded-tl-none p-4 text-sm text-gray-300 leading-relaxed shadow-sm">
                            <p>System Online. I am analyzing the <span class="text-white font-bold">XAUUSD</span> order flow. Latency is 12ms. What's your play?</p>
                        </div>
                    </div>
                </div>

                <template x-for="msg in messages" :key="msg.id">
                    <div class="flex items-start gap-4 animate-fade-in-up" :class="msg.isUser ? 'flex-row-reverse' : ''">
                        <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg"
                             :class="msg.isUser ? 'bg-white/10' : 'bg-gradient-to-br from-cyan-500 to-blue-600 shadow-cyan-500/20'">
                             <template x-if="msg.isUser">
                                <svg class="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                             </template>
                             <template x-if="!msg.isUser">
                                <svg class="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                             </template>
                        </div>
                        <div class="space-y-1 text-right" :class="msg.isUser ? 'items-end' : 'text-left items-start'">
                            <span class="text-[10px] font-bold ml-1" :class="msg.isUser ? 'text-gray-500 mr-1' : 'text-cyan-400'" x-text="msg.isUser ? 'YOU' : 'PIPFLOW'"></span>
                            <div class="p-4 text-sm leading-relaxed shadow-sm max-w-[280px]"
                                 :class="msg.isUser ? 'bg-violet-600 text-white rounded-2xl rounded-tr-none shadow-violet-500/20' : 'bg-white/5 border border-white/5 text-gray-300 rounded-2xl rounded-tl-none'">
                                <p x-text="msg.text"></p>
                            </div>
                        </div>
                    </div>
                </template>
                
                <div x-show="isTyping" class="flex items-start gap-4 animate-pulse">
                    <div class="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center flex-shrink-0">
                        <svg class="w-4 h-4 text-cyan-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z" /></svg>
                    </div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="px-4 py-2 border-t border-white/5 bg-[#0a0a0a]">
                <div class="flex gap-2 overflow-x-auto custom-scrollbar pb-2">
                    <button @click="input = 'Analyze this market structure'; sendMessage()" class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs font-medium text-gray-400 hover:text-cyan-400 hover:bg-cyan-400/10 hover:border-cyan-400/20 whitespace-nowrap transition-all">
                        Analyze Market
                    </button>
                    <button @click="input = 'Suggest a strategy for this trend'; sendMessage()" class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs font-medium text-gray-400 hover:text-violet-400 hover:bg-violet-400/10 hover:border-violet-400/20 whitespace-nowrap transition-all">
                        Create Strategy
                    </button>
                    <button @click="input = 'Assess risk for 1 lot trade'; sendMessage()" class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs font-medium text-gray-400 hover:text-emerald-400 hover:bg-emerald-400/10 hover:border-emerald-400/20 whitespace-nowrap transition-all">
                        Risk Check
                    </button>
                    <button @click="input = 'Buy 0.1 Lot'; sendMessage()" class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs font-medium text-gray-400 hover:text-green-400 hover:bg-green-400/10 hover:border-green-400/20 whitespace-nowrap transition-all">
                        Buy 0.1
                    </button>
                    <button @click="input = 'Sell 0.1 Lot'; sendMessage()" class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 text-xs font-medium text-gray-400 hover:text-red-400 hover:bg-red-400/10 hover:border-red-400/20 whitespace-nowrap transition-all">
                        Sell 0.1
                    </button>
                </div>
            </div>

            <!-- Input Area -->
            <div class="p-4 bg-[#0a0a0a] border-t border-white/5">
                <form @submit.prevent="sendMessage" class="relative group">
                    <div class="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition-opacity"></div>
                    <input type="text" x-model="input" placeholder="Ask PipFlow..." 
                        class="w-full bg-[#111] border border-white/10 rounded-xl pl-4 pr-12 py-3.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-violet-500/50 focus:bg-[#151515] transition-all relative z-10">
                    <button type="submit" 
                        class="absolute right-2 top-2 p-1.5 rounded-lg bg-white/5 text-gray-400 hover:text-white hover:bg-violet-500 hover:shadow-lg hover:shadow-violet-500/50 transition-all z-20 disabled:opacity-50 disabled:cursor-not-allowed" 
                        :disabled="!input.trim()">
                        <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                    </button>
                </form>
            </div>
        </div>

    </div>

    <!-- Chart Engine Script -->
    <script>
        document.addEventListener('alpine:init', () => {
             Alpine.data('alphaTerminal', () => {
                 // Closure variables to avoid Alpine Proxy issues with complex objects
                 let chart = null;
                 let candleSeries = null;

                 return {
                    activeTimeframe: '1H',
                    activeAsset: 'XAUUSD',
                    fullscreen: false,
                    status: { loading: true, error: false, message: '', debug: 'Waiting for libs...' },
                    assets: {
                        'XAUUSD': { name: 'Gold Spot', price: 2034.50 },
                        'BTCUSD': { name: 'Bitcoin', price: 64200.00 },
                        'EURUSD': { name: 'Euro/USD', price: 1.0850 },
                    },
                    messages: [],
                    input: '',
                    isTyping: false,
                    isScanning: false, // New State
                    activePriceLines: [], // To track and remove old TP/SL lines

                    // NO reactive chart properties here

                    init() {
                        this.waitForLib();
                    },

                    // Trigger the Scan Effect & Analysis
                    scanMarket() {
                        this.isScanning = true;
                        this.messages.push({ id: Date.now(), text: "Initiating Deep Market Scan (Order Flow + Volume Profile)...", isUser: false });
                        
                        // Clear old lines
                        if(this.activePriceLines.length > 0 && candleSeries) {
                            this.activePriceLines.forEach(line => candleSeries.removePriceLine(line));
                            this.activePriceLines = [];
                        }
                        if(candleSeries) candleSeries.setMarkers([]); // Clear markers

                        // Simulate AI processing time matching the animation
                        setTimeout(() => {
                            this.isScanning = false;
                            this.drawTradeSetup();
                        }, 5000);
                    },

                    drawTradeSetup() {
                         const currentPrice = this.assets[this.activeAsset].price;
                         const direction = Math.random() > 0.5 ? 'Buy' : 'Sell';
                         const offset = this.activeAsset === 'XAUUSD' ? 2.5 : 0.0020; // Example spread adjustment
                         
                         const sl = direction === 'Buy' ? currentPrice - offset : currentPrice + offset;
                         const tp = direction === 'Buy' ? currentPrice + (offset * 2) : currentPrice - (offset * 2);

                         // 1. Add Markers (Arrows)
                         // We need a time for the marker. Let's use the last known mock candle time or just 'now'
                         // Since we don't have exact candle object here easily without querying series, 
                         // we will just assume the setup is 'Latest'. 
                         // *Note: Markers require 'time' matching a bar. 
                         // For this simulation, we'll fetch the last data point time from our internal generated data or just skip if complex.*
                         // Simpler approach: Just use Price Lines which are time-independent.
                         
                         if (!candleSeries) return;

                         // Draw TP Line
                         const tpLine = candleSeries.createPriceLine({
                            price: tp,
                            color: '#10b981',
                            lineWidth: 2,
                            lineStyle: 0, // Solid
                            axisLabelVisible: true,
                            title: 'TP (AI Target)',
                         });
                         this.activePriceLines.push(tpLine);

                         // Draw SL Line
                         const slLine = candleSeries.createPriceLine({
                            price: sl,
                            color: '#ef4444',
                            lineWidth: 2,
                            lineStyle: 0, // Solid
                            axisLabelVisible: true,
                            title: 'SL (Protection)',
                         });
                         this.activePriceLines.push(slLine);

                         // Draw Entry Marker (Mocking time roughly)
                         // Getting last bar time is tricky from outside without tracking it. 
                         // We will rely on text feedback mostly, but lines are great.

                         this.messages.push({ 
                            id: Date.now(), 
                            text: `Scan Complete. \nStrategy: ${direction} Impulsive \n✅ TP: ${this.formatPrice(tp)} \n🛑 SL: ${this.formatPrice(sl)} \nConfidence: 87%`, 
                            isUser: false 
                         });
                    },

                    waitForLib() {
                         if(window.LightweightCharts && typeof window.LightweightCharts.createChart === 'function') {
                             this.status.debug = 'Library Loaded'; 
                             this.initChart();
                             this.generateMockData();
                             this.status.loading = false;
                         } else {
                             // Try to load via CDN if local failed
                             if(!document.getElementById('lw-cdn')) {
                                 this.status.debug = 'Local failed, trying CDN (v4.2.0)...';
                                 const script = document.createElement('script');
                                 script.id = 'lw-cdn';
                                 // PINNED VERSION 4.2.0 to ensure API compatibility (v5 removed helper methods)
                                 script.src = 'https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js';
                                 script.onload = () => { this.waitForLib(); };
                                 document.head.appendChild(script);
                             } else {
                                 this.status.debug = 'Waiting for LightweightCharts CDN...';
                                 setTimeout(() => this.waitForLib(), 500);
                             }
                         }
                    },

                    formatPrice(price) {
                        return new Intl.NumberFormat('en-US', { minimumFractionDigits: 2 }).format(price);
                    },

                    switchAsset(symbol) {
                        this.activeAsset = symbol;
                        this.messages.push({
                            id: Date.now(),
                            text: `Switching analysis to ${symbol}. Loading technicals...`,
                            isUser: false
                        });
                        this.generateMockData();
                    },

                    setTimeframe(tf) {
                        this.activeTimeframe = tf;
                        this.generateMockData();
                    },

                    initChart() {
                        const chartContainer = document.getElementById('alphaChart');
                        if(!window.LightweightCharts) return;

                        // Clear if exists
                        if(chart) {
                            chart.remove();
                            chartContainer.innerHTML = '';
                        }

                        chart = window.LightweightCharts.createChart(chartContainer, {
                            layout: {
                                background: { color: '#050505' },
                                textColor: '#6b7280',
                                fontFamily: 'Inter, sans-serif',
                            },
                            grid: {
                                vertLines: { color: '#ffffff05' },
                                horzLines: { color: '#ffffff05' },
                            },
                            crosshair: {
                                mode: 1, // Magnet
                                vertLine: { color: '#8b5cf6', width: 1, style: 3, labelBackgroundColor: '#8b5cf6' },
                                horzLine: { color: '#8b5cf6', width: 1, style: 3, labelBackgroundColor: '#8b5cf6' },
                            },
                            timeScale: { borderColor: '#ffffff10' },
                            rightPriceScale: { borderColor: '#ffffff10' },
                        });

                        candleSeries = chart.addCandlestickSeries({
                            upColor: '#10b981',
                            downColor: '#ef4444', 
                            borderVisible: false,
                            wickUpColor: '#10b981',
                            wickDownColor: '#ef4444',
                        });

                        new ResizeObserver(entries => {
                            if (!chartContainer || !chart) return;
                            for (let entry of entries) {
                                const { width, height } = entry.contentRect;
                                chart.applyOptions({ width, height });
                            }
                        }).observe(chartContainer);
                    },

                    generateMockData() {
                        fetch(`/api/trading/history/${this.activeAsset}?timeframe=${this.activeTimeframe}`)
                            .then(response => response.json())
                            .then(data => {
                                 if(data.success && candleSeries) {
                                     // Ensure data is sorted by time
                                     const sortedData = data.data.sort((a, b) => a.time - b.time);
                                     candleSeries.setData(sortedData);
                                     chart.timeScale().fitContent();
                                 }
                            })
                            .catch(error => console.error('Error fetching data:', error));
                    },

                    sendMessage() {
                        if (!this.input.trim()) return;
                        this.messages.push({ id: Date.now(), text: this.input, isUser: true });
                        const q = this.input.toLowerCase();
                        this.input = '';
                        this.scrollToBottom();

                        if (q.includes('analyze') || q.includes('analysis')) {
                            this.scanMarket();
                            return;
                        }

                        this.isTyping = true;
                        
                        setTimeout(() => {
                            this.isTyping = false;
                            let r = "I'm calculating optimal entries driven by recent volume spikes.";
                            if(q.includes('buy')) r = "Bullish momentum detected. RSI at 45. Consider long entries above 2038.00.";
                            if(q.includes('sell')) r = "Bearish divergence on H4. Pivot point at 2040.00 acts as strong resistance.";
                            this.messages.push({ id: Date.now(), text: r, isUser: false });
                            this.scrollToBottom();
                        }, 1000 + Math.random() * 1000);
                    },

                    scrollToBottom() {
                        this.$nextTick(() => {
                            this.$refs.chatStream.scrollTop = this.$refs.chatStream.scrollHeight;
                        });
                    }
                 };
             });
        });
    </script>
    
    
    <style>
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #555; }
        
        @keyframes scanPremium {
            0% { top: 0%; opacity: 0; }
            10% { opacity: 1; }
            50% { top: 100%; opacity: 1; }
            90% { opacity: 1; }
            100% { top: 0%; opacity: 0; }
        }
    </style>
</x-app-layout>
