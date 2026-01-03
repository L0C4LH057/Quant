<x-app-layout>
    <div x-data="{ 
            accounts: {{ $accounts->map(fn($a) => ['id' => $a->id, 'login' => $a->login, 'server' => $a->server, 'balance' => $a->balance, 'is_ai_active' => $a->is_ai_active])->toJson() }},
            selectedAccountId: null,
            aiActive: false,
            logs: [],
            get currentAccount() {
                if (this.accounts.length === 0) return null;
                return this.accounts.find(a => a.id === this.selectedAccountId) || this.accounts[0];
            },
            formatMoney(amount) {
                return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
            },
            init() {
                if (this.accounts.length > 0) {
                    this.selectedAccountId = this.accounts[0].id;
                    this.aiActive = !!this.accounts[0].is_ai_active;
                }
                this.renderChart();
                this.startTerminal();

                this.$watch('currentAccount', (val) => {
                    if(val) {
                        this.aiActive = !!val.is_ai_active;
                    }
                });
            },
            toggleAi() {
                console.log('Toggle AI clicked. Current Account:', this.currentAccount);
                if(!this.currentAccount) {
                    console.error('No current account selected.');
                    return;
                }

                // Optimistic UI update
                this.aiActive = !this.aiActive;
                console.log('Optimistic AI State:', this.aiActive);
                
                fetch(`/dashboard/toggle-ai/${this.currentAccount.id}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-TOKEN': '{{ csrf_token() }}'
                    }
                })
                .then(res => {
                    if (!res.ok) throw new Error('Network response was not ok');
                    return res.json();
                })
                .then(data => {
                    console.log('Server Response:', data);
                    if(data.success) {
                        // Sync truth
                        this.currentAccount.is_ai_active = data.is_ai_active; 
                        this.aiActive = !!data.is_ai_active;
                        this.addLog(data.message, this.aiActive ? 'text-green-400' : 'text-gray-400');
                    } else {
                         console.error('Server returned success: false');
                         this.aiActive = !this.aiActive; // Revert
                    }
                })
                .catch(error => {
                    console.error('Error toggling AI:', error);
                    this.aiActive = !this.aiActive; // Revert on error
                    this.addLog('Connection Error: Failed to toggle AI.', 'text-red-500');
                });
            },
            addLog(message, color = 'text-gray-400') {
                const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: 'numeric', minute: 'numeric', second: 'numeric' });
                this.logs.push({ id: Date.now(), time: `[${time}]`, message: message, color: color });
                this.$nextTick(() => {
                    if(this.$refs.terminal) this.$refs.terminal.scrollTop = this.$refs.terminal.scrollHeight;
                });
            },
            startTerminal() {
                const messages = [
                    { msg: 'Scanning market structure for XAUUSD...', color: 'text-blue-400' },
                    { msg: 'Detected liquidity void at 2035.50.', color: 'text-yellow-500' },
                    { msg: 'Analyzing Order Flow...', color: 'text-gray-400' },
                    { msg: 'RSI divergence confirmation on 15m timeframe.', color: 'text-green-400' },
                    { msg: 'Waiting for entry confirmation...', color: 'text-gray-400' }
                ];
                
                let i = 0;
                this.addLog('PipFlow Core System v2.1 Initialized.', 'text-brand-500');
                
                setInterval(() => {
                    if(this.aiActive && this.currentAccount) {
                        const item = messages[Math.floor(Math.random() * messages.length)];
                        this.addLog(item.msg, item.color);
                    } else if (this.aiActive && !this.currentAccount) {
                        this.addLog('Waiting for Broker Connection...', 'text-red-400');
                    }
                }, 3500);
            },
            renderChart() {
                if(!window.ApexCharts) return;
                
                const options = {
                    series: [{
                        name: 'Equity',
                        data: this.currentAccount ? [this.currentAccount.balance * 0.9, this.currentAccount.balance] : [0, 0]
                    }],
                    chart: {
                        type: 'area',
                        height: 300,
                        toolbar: { show: false },
                        background: 'transparent',
                        fontFamily: 'Nunito, sans-serif'
                    },
                    colors: ['#f59e0b'],
                    fill: {
                        type: 'gradient',
                        gradient: { shadeIntensity: 1, opacityFrom: 0.4, opacityTo: 0.05, stops: [0, 90, 100] }
                    },
                    dataLabels: { enabled: false },
                    stroke: { curve: 'smooth', width: 2 },
                    xaxis: {
                        categories: ['Start', 'Current'],
                        axisBorder: { show: false }, axisTicks: { show: false }, labels: { style: { colors: '#666' } }
                    },
                    yaxis: { labels: { style: { colors: '#666' } } },
                    grid: { borderColor: '#333', strokeDashArray: 4, yaxis: { lines: { show: true } }, xaxis: { lines: { show: false } } },
                    theme: { mode: 'dark' }
                };

                const chart = new ApexCharts(document.querySelector('#equityChart'), options);
                chart.render();
                
                this.$watch('currentAccount', (val) => {
                    if(val) {
                        chart.updateSeries([{ data: [val.balance * 0.95, val.balance] }]);
                    } else {
                         chart.updateSeries([{ data: [0, 0] }]);
                    }
                });
            }
         }">
        <!-- Header -->
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
            <div>
                <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">Command Center</h1>
                <p class="text-gray-500 text-sm mt-1">Welcome back, {{ Auth::user()->name }}</p>
            </div>
            
            <div class="flex items-center gap-4">
                
                <!-- Account Selector -->
                <div class="relative" x-data="{ open: false }" x-show="accounts.length > 0">
                    <button @click="open = !open" @click.away="open = false" class="flex items-center gap-3 px-4 py-2 bg-[#0a0a0a] border border-white/10 rounded-xl hover:bg-white/5 transition-all min-w-[200px] justify-between group">
                        <div class="flex items-center gap-3">
                            <div class="w-8 h-8 rounded-full bg-brand-500/10 flex items-center justify-center">
                                <svg class="w-4 h-4 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
                            </div>
                            <div class="text-left">
                                <p class="text-[10px] text-gray-500" x-text="currentAccount ? currentAccount.server : 'Broker'"></p>
                                <p class="text-sm font-bold text-white" x-text="currentAccount ? currentAccount.login : 'Select Account'"></p>
                            </div>
                        </div>
                        <svg class="w-4 h-4 text-gray-500 group-hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
                    </button>

                    <div x-show="open" 
                         class="absolute top-full right-0 mt-2 w-64 bg-[#0a0a0a] border border-white/10 rounded-xl shadow-2xl py-2 z-50">
                        <template x-for="acc in accounts" :key="acc.id">
                            <button @click="selectedAccountId = acc.id; open = false" 
                                class="w-full px-4 py-3 text-left hover:bg-white/5 transition-colors flex items-center justify-between group border-b border-white/5 last:border-0">
                                <div class="flex items-center gap-3">
                                    <span class="w-2 h-2 rounded-full" :class="selectedAccountId === acc.id ? 'bg-brand-500' : 'bg-gray-700'"></span>
                                    <div>
                                        <p class="text-sm font-bold text-white group-hover:text-brand-400" x-text="acc.login"></p>
                                        <p class="text-[10px] text-gray-500" x-text="acc.server"></p>
                                    </div>
                                </div>
                                <span class="text-xs font-mono text-gray-400" x-text="formatMoney(acc.balance)"></span>
                            </button>
                        </template>
                        <a href="{{ route('settings.index') }}" class="block w-full px-4 py-3 text-center text-xs text-brand-500 hover:text-brand-400 font-bold border-t border-white/10">
                            + Connect New Account
                        </a>
                    </div>
                </div>

                <!-- No Accounts State -->
                <div x-show="accounts.length === 0">
                    <a href="{{ route('settings.index') }}" class="flex items-center gap-2 px-4 py-2 bg-brand-500 text-black text-sm font-bold rounded-xl hover:bg-brand-400 transition-colors shadow-[0_0_15px_rgba(234,179,8,0.3)]">
                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>
                        Connect Broker
                    </a>
                </div>

                <!-- AI Toggle -->
                <button @click="toggleAi" 
                    class="flex items-center gap-3 px-5 py-2.5 rounded-xl border transition-all duration-300 shadow-lg"
                    :class="aiActive ? 'bg-brand-500/10 border-brand-500 text-brand-400 shadow-brand-500/10' : 'bg-red-500/10 border-red-500/50 text-red-400'">
                    <div class="relative w-3 h-3">
                        <div class="absolute inset-0 rounded-full animate-ping opacity-75" :class="aiActive ? 'bg-brand-500' : 'bg-red-500'"></div>
                        <div class="relative w-3 h-3 rounded-full" :class="aiActive ? 'bg-brand-500' : 'bg-red-500'"></div>
                    </div>
                    <span class="font-bold tracking-wide" x-text="aiActive ? 'AI ACTIVE' : 'AI DORMANT'"></span>
                </button>

                <!-- User Profile Dropdown -->
                <div class="relative" x-data="{ open: false }">
                    <button @click="open = !open" @click.away="open = false" class="flex items-center gap-3 pl-4 border-l border-white/10 focus:outline-none group">
                        <div class="text-right hidden md:block">
                            <p class="text-sm font-bold text-white group-hover:text-brand-400 transition-colors">{{ Auth::user()->name }}</p>
                            @if(Auth::user()->subscription_plan === 'free')
                                <p class="text-xs text-gray-500">Free Plan</p>
                            @else
                                <p class="text-xs text-brand-500 font-bold uppercase tracking-wider">{{ str_replace('_', ' ', Auth::user()->subscription_plan) }}</p>
                            @endif
                        </div>
                        <div class="w-10 h-10 rounded-full bg-gradient-to-br from-brand-400 to-brand-600 p-[1px] shadow-lg shadow-brand-500/20">
                            <div class="w-full h-full rounded-full bg-[#0a0a0a] flex items-center justify-center">
                                <span class="font-bold text-brand-500">{{ substr(Auth::user()->name, 0, 2) }}</span>
                            </div>
                        </div>
                    </button>

                    <!-- Dropdown Menu -->
                    <div x-show="open" 
                         x-transition:enter="transition ease-out duration-200"
                         x-transition:enter-start="opacity-0 translate-y-2"
                         x-transition:enter-end="opacity-100 translate-y-0"
                         x-transition:leave="transition ease-in duration-150"
                         x-transition:leave-start="opacity-100 translate-y-0"
                         x-transition:leave-end="opacity-0 translate-y-2"
                         class="absolute right-0 mt-3 w-56 rounded-xl bg-[#0a0a0a] border border-white/10 shadow-2xl py-2 z-50" 
                         style="display: none;">
                        
                        <div class="px-4 py-3 border-b border-white/5 mb-2">
                            <p class="text-sm text-gray-400">Signed in as</p>
                            <p class="text-sm font-bold text-white truncate">{{ Auth::user()->email }}</p>
                        </div>

                        <a href="{{ route('profile.edit') }}" class="block px-4 py-2.5 text-sm text-gray-300 hover:bg-white/5 hover:text-brand-400 transition-colors flex items-center gap-2">
                            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                            User Profile
                        </a>

                        @if(Auth::user()->subscription_plan === 'free')
                            <a href="{{ route('subscription.index') }}" class="block px-4 py-2.5 mx-2 my-1 text-center rounded-lg bg-gradient-to-r from-brand-400 to-brand-600 text-black text-xs font-bold uppercase tracking-wider hover:opacity-90 transition-opacity flex items-center justify-center gap-2">
                                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                                Go Premium
                            </a>
                        @else
                            <a href="{{ route('subscription.manage') }}" class="block px-4 py-2.5 text-sm text-brand-400 hover:bg-white/5 transition-colors flex items-center gap-2">
                                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
                                Manage Subscription
                            </a>
                        @endif
                        
                        <a href="{{ route('settings.index') }}" class="block px-4 py-2.5 text-sm text-gray-300 hover:bg-white/5 hover:text-brand-400 transition-colors flex items-center gap-2">
                            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                            Settings
                        </a>

                        <div class="border-t border-white/5 my-2"></div>

                        <form method="POST" action="{{ route('logout') }}">
                            @csrf
                            <button type="submit" class="w-full text-left px-4 py-2.5 text-sm text-red-400 hover:bg-red-500/10 transition-colors flex items-center gap-2">
                                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 11-6 0v-1m6 0H9" /></svg>
                                Sign Out
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <!-- Widgets Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            
            <!-- Card 1: Total Equity -->
            <div class="p-6 rounded-2xl bg-[#0a0a0a]/80 backdrop-blur border border-white/5 relative overflow-hidden group">
                <div class="absolute top-0 right-0 w-32 h-32 bg-brand-500/10 rounded-full blur-[50px] group-hover:bg-brand-500/20 transition-all"></div>
                <h3 class="text-gray-500 text-sm font-medium mb-1">Total Equity</h3>
                <div class="flex items-baseline gap-2">
                    <span class="text-3xl font-bold text-white tracking-tight" x-text="currentAccount ? formatMoney(currentAccount.balance) : '$0.00'"></span>
                    <span class="text-sm text-green-400 bg-green-400/10 px-2 py-0.5 rounded-lg" x-text="currentAccount ? '+4.2%' : '0%'"></span>
                </div>
            </div>

            <!-- Card 2: Net Profit -->
            <div class="p-6 rounded-2xl bg-[#0a0a0a]/80 backdrop-blur border border-white/5 hover:border-green-500/20 transition-all group">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-gray-500 text-sm font-medium">Net Profit</h3>
                    <div class="p-1.5 rounded-lg bg-green-500/10 text-green-500 group-hover:bg-green-500/20 transition-colors">
                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold text-green-400">+${{ number_format($aiStats['total_profit'], 2) }}</span>
                    <span class="text-xs text-green-500/80 bg-green-500/10 px-1.5 py-0.5 rounded">PF {{ $aiStats['profit_factor'] }}</span>
                </div>
            </div>

            <!-- Card 3: Total Loss -->
            <div class="p-6 rounded-2xl bg-[#0a0a0a]/80 backdrop-blur border border-white/5 hover:border-red-500/20 transition-all group">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-gray-500 text-sm font-medium">Total Loss</h3>
                    <div class="p-1.5 rounded-lg bg-red-500/10 text-red-500 group-hover:bg-red-500/20 transition-colors">
                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" /></svg>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold text-red-400">-${{ number_format($aiStats['total_loss'], 2) }}</span>
                    <span class="text-xs text-red-500/80">Realized</span>
                </div>
            </div>

            <!-- Card 4: Total Trades -->
            <div class="p-6 rounded-2xl bg-[#0a0a0a]/80 backdrop-blur border border-white/5 hover:border-blue-500/20 transition-all group">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-gray-500 text-sm font-medium">Total AI Trades</h3>
                    <div class="p-1.5 rounded-lg bg-blue-500/10 text-blue-500 group-hover:bg-blue-500/20 transition-colors">
                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                    </div>
                </div>
                <div class="flex items-baseline gap-2">
                    <span class="text-2xl font-bold text-white">{{ $aiStats['total_trades'] }}</span>
                    <span class="text-xs text-blue-500">{{ $aiStats['win_rate'] }}% Win Rate</span>
                </div>
            </div>
        </div>

        <!-- Main Content Split -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            <!-- Left Column: Trade History / Open Positions (8 Cols) -->
            <div class="lg:col-span-8 p-6 rounded-2xl bg-[#0a0a0a]/80 backdrop-blur border border-white/5 h-fit" x-data="{ tab: 'history' }">
                <div class="flex items-center justify-between mb-6">
                    <h3 class="font-bold text-lg text-white" x-text="tab === 'history' ? 'AI Trade History' : 'Open Positions'"></h3>
                    <div class="flex gap-2">
                         <button @click="tab = 'history'" 
                             :class="tab === 'history' ? 'bg-white/10 text-white' : 'text-gray-500 hover:bg-white/5'"
                             class="px-3 py-1 text-xs font-medium rounded-lg transition-colors">History</button>
                         <button @click="tab = 'positions'" 
                             :class="tab === 'positions' ? 'bg-white/10 text-white' : 'text-gray-500 hover:bg-white/5'"
                             class="px-3 py-1 text-xs font-medium rounded-lg transition-colors">Open Positions</button>
                    </div>
                </div>
                
                <div class="overflow-x-auto">
                    <table class="w-full text-left">
                        <thead>
                            <tr class="text-xs text-gray-500 border-b border-white/5">
                                <th class="pb-3 pl-2 font-medium">PAIR</th>
                                <th class="pb-3 font-medium">TYPE</th>
                                <th class="pb-3 font-medium" x-text="tab === 'history' ? 'PRICE' : 'ENTRY'"></th>
                                <th class="pb-3 font-medium">PROFIT/LOSS</th>
                                <th class="pb-3 pr-2 text-right font-medium">TIME</th>
                            </tr>
                        </thead>
                        <tbody class="text-sm">
                            <!-- History Rows -->
                            <template x-if="tab === 'history'">
                                @if($aiTrades->count() > 0)
                                    @foreach($aiTrades as $trade)
                                    <tr class="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors group">
                                        <td class="py-4 pl-2 font-bold text-white">{{ $trade['pair'] }}</td>
                                        <td class="py-4">
                                            <span class="px-2 py-1 rounded text-[10px] font-bold uppercase {{ $trade['type'] === 'BUY' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500' }}">
                                                {{ $trade['type'] }}
                                            </span>
                                        </td>
                                        <td class="py-4 text-gray-400">{{ number_format($trade['price'], 5) }}</td>
                                        <td class="py-4 font-bold {{ $trade['profit'] >= 0 ? 'text-green-400' : 'text-red-400' }}">
                                            {{ $trade['profit'] >= 0 ? '+' : '' }}${{ number_format($trade['profit'], 2) }}
                                        </td>
                                        <td class="py-4 pr-2 text-right text-gray-500 text-xs">{{ $trade['time']->diffForHumans() }}</td>
                                    </tr>
                                    @endforeach
                                @else
                                    <tr>
                                        <td colspan="5" class="py-8 text-center text-gray-500 text-sm">No recent history found.</td>
                                    </tr>
                                @endif
                            </template>

                            <!-- Open Positions Rows -->
                            <template x-if="tab === 'positions'">
                                @if($openPositions->count() > 0)
                                    @foreach($openPositions as $pos)
                                    <tr class="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors group">
                                        <td class="py-4 pl-2 font-bold text-white">{{ $pos['pair'] }}</td>
                                        <td class="py-4">
                                            <span class="px-2 py-1 rounded text-[10px] font-bold uppercase {{ $pos['type'] === 'BUY' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500' }}">
                                                {{ $pos['type'] }}
                                            </span>
                                            <span class="text-xs text-gray-500 ml-1">{{ $pos['volume'] }}</span>
                                        </td>
                                        <td class="py-4 text-gray-400">
                                            <div>{{ number_format($pos['entry_price'], 5) }}</div>
                                            <div class="text-[10px] text-gray-600">{{ number_format($pos['current_price'], 5) }}</div>
                                        </td>
                                        <td class="py-4 font-bold {{ $pos['profit'] >= 0 ? 'text-green-400' : 'text-red-400' }}">
                                            {{ $pos['profit'] >= 0 ? '+' : '' }}${{ number_format($pos['profit'], 2) }}
                                        </td>
                                        <td class="py-4 pr-2 text-right text-gray-500 text-xs">{{ $pos['time']->diffForHumans() }}</td>
                                    </tr>
                                    @endforeach
                                @else
                                    <tr>
                                        <td colspan="5" class="py-8 text-center text-gray-500 text-sm">No open positions.</td>
                                    </tr>
                                @endif
                            </template>
                        </tbody>
                    </table>
                </div>
                <div class="mt-4 pt-4 border-t border-white/5 text-center" x-show="tab === 'history'">
                    <a href="#" class="text-xs text-gray-500 hover:text-brand-400 transition-colors">View All Transaction History &rarr;</a>
                </div>
            </div>

            <!-- Right Column: AI Live Terminal (4 Cols) -->
            <div class="lg:col-span-4 flex flex-col gap-6">
                <!-- Live Terminal -->
                <div class="p-6 rounded-2xl bg-[#0a0a0a] border border-brand-500/20 shadow-[0_0_30px_rgba(245,158,11,0.05)] relative overflow-hidden font-mono text-sm leading-relaxed flex-1 min-h-[400px]">
                    <div class="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-transparent via-brand-500 to-transparent opacity-50"></div>
                    
                    <h3 class="text-brand-400 font-bold mb-4 flex items-center gap-2">
                        <span class="w-2 h-2 bg-brand-500 rounded-full animate-pulse"></span>
                        AI Live System Log
                    </h3>
                    
                    <div class="space-y-3 h-[350px] overflow-y-auto custom-scrollbar pr-2" x-ref="terminal">
                        <template x-for="log in logs" :key="log.id">
                            <div class="flex gap-2 text-xs">
                                <span class="text-gray-600 shrink-0" x-text="log.time"></span>
                                <span :class="log.color" x-text="log.message" class="break-words"></span>
                            </div>
                        </template>
                        <div class="flex gap-2 animate-pulse">
                            <span class="text-gray-600">></span>
                            <span class="text-brand-500">_</span>
                        </div>
                    </div>
                </div>

                <!-- Quick Action / Status -->
                <div class="p-6 rounded-2xl bg-[#0a0a0a]/80 backdrop-blur border border-white/5 flex items-center justify-between">
                     <div>
                        <p class="text-xs text-gray-500 uppercase tracking-wider font-bold">System Status</p>
                        <p class="text-white font-bold flex items-center gap-2 mt-1">
                            <span class="w-2 h-2 rounded-full bg-green-500"></span>
                            Operational
                        </p>
                     </div>
                     <div class="text-right">
                        <p class="text-xs text-gray-500 uppercase tracking-wider font-bold">Latency</p>
                        <p class="text-brand-400 font-bold mt-1">24ms</p>
                     </div>
                </div>
            </div>

        </div>    </div>


</x-app-layout>
