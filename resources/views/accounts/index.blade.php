<x-app-layout>
    <div x-data="{ activeTab: 'mt5' }" class="p-6 max-w-7xl mx-auto space-y-8">
        
        <!-- Headers -->
        <div>
            <h2 class="text-3xl font-bold text-white">Trading Accounts</h2>
            <p class="text-gray-500 mt-2">Manage your connected brokers and exchanges.</p>
        </div>

        @if($errors->has('sync'))
            <div class="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm font-bold flex items-center gap-2">
                <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                {{ $errors->first('sync') }}
            </div>
        @endif

        @if(session('status'))
            <div class="p-4 rounded-xl bg-green-500/10 border border-green-500/20 text-green-400">
                {{ session('status') }}
            </div>
        @endif

        <div class="space-y-4">
            <h3 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="w-2 h-6 bg-brand-500 rounded-full"></span>
                Connected Accounts
            </h3>

            @if($accounts->count() > 0)
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    @foreach($accounts as $account)
                        <div class="p-5 rounded-2xl bg-[#0a0a0a] border border-white/5 flex flex-col justify-between group hover:border-brand-500/30 transition-all">
                            <div class="flex items-start justify-between mb-4">
                                <div class="flex items-center gap-3">
                                    <div class="w-10 h-10 rounded-lg {{ $account->broker_type == 'deriv' ? 'bg-red-500/10 text-red-500' : 'bg-blue-500/10 text-blue-500' }} flex items-center justify-center">
                                        @if($account->broker_type == 'deriv')
                                            <svg class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                                        @else
                                            <svg class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
                                        @endif
                                    </div>
                                    <div>
                                        <h4 class="font-bold text-white capitalize">{{ $account->name ?? ($account->broker_type == 'mt5' ? 'MetaTrader 5' : ($account->broker_type == 'mt4' ? 'MetaTrader 4' : 'Deriv')) }}</h4>
                                        <p class="text-xs text-gray-500">{{ $account->login_id }}</p>
                                    </div>
                                </div>
                                <div class="flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-500/10 border border-green-500/20">
                                    <span class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                                    <span class="text-xs font-bold text-green-500 uppercase tracking-wider">Active</span>
                                </div>
                            </div>
                            
                            <div class="flex items-end justify-between border-t border-white/5 pt-4">
                                    <div>
                                    <p class="text-xs text-gray-500 mb-1">Balance</p>
                                    <p class="text-xl font-bold text-white">${{ number_format($account->balance, 2) }}</p>
                                </div>
                                    <button class="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-colors">
                                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"></path></svg>
                                    </button>
                            </div>
                        </div>
                    @endforeach
                </div>
            @else
                <div class="p-8 rounded-2xl bg-[#0a0a0a] border border-white/5 border-dashed text-center">
                    <p class="text-gray-500">No accounts connected yet. Connect one below to start trading.</p>
                </div>
            @endif
        </div>

        <!-- Section 2: Connection Tabs -->
        <div class="space-y-6">
            <h3 class="text-xl font-bold text-white flex items-center gap-2">
                <span class="w-2 h-6 bg-gray-500 rounded-full"></span>
                Connect New Account
            </h3>

            <div class="bg-[#0a0a0a] border border-white/5 rounded-2xl overflow-hidden">
                <!-- Tabs Header -->
                <div class="flex border-b border-white/5">
                    <button @click="activeTab = 'mt5'" :class="activeTab === 'mt5' ? 'border-brand-500 text-brand-400 bg-brand-500/5' : 'border-transparent text-gray-500 hover:text-white hover:bg-white/5'" class="flex-1 py-4 text-sm font-bold uppercase tracking-wider border-b-2 transition-all duration-200">
                        MetaTrader 4/5
                    </button>
                    <button @click="activeTab = 'deriv'" :class="activeTab === 'deriv' ? 'border-red-500 text-red-400 bg-red-500/5' : 'border-transparent text-gray-500 hover:text-white hover:bg-white/5'" class="flex-1 py-4 text-sm font-bold uppercase tracking-wider border-b-2 transition-all duration-200">
                        Deriv
                    </button>
                    <button @click="activeTab = 'crypto'" :class="activeTab === 'crypto' ? 'border-purple-500 text-purple-400 bg-purple-500/5' : 'border-transparent text-gray-500 hover:text-white hover:bg-white/5'" class="flex-1 py-4 text-sm font-bold uppercase tracking-wider border-b-2 transition-all duration-200">
                        Crypto Exchange
                    </button>
                </div>

                <!-- Tab Contents -->
                <div class="p-6 md:p-8">
                    
                    <!-- MT5 Tab -->
                    <div x-show="activeTab === 'mt5'" x-transition:enter="transition ease-out duration-300" x-transition:enter-start="opacity-0 translate-y-2" x-transition:enter-end="opacity-100 translate-y-0">
                        <form action="{{ route('accounts.connect') }}" method="POST" class="max-w-2xl mx-auto space-y-5">
                            @csrf
                            <input type="hidden" name="type" value="mt5">
                            
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                                <div class="md:col-span-2">
                                    <label class="block text-sm font-medium text-gray-400 mb-1">Server</label>
                                    <label class="block text-sm font-medium text-gray-400 mb-1">Server</label>
                                    <input type="text" name="server" class="w-full bg-[#050505] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors" placeholder="e.g. MetaQuotes-Demo">
                                </div>
                                
                                <div>
                                    <label class="block text-sm font-medium text-gray-400 mb-1">Login ID</label>
                                    <input type="text" name="login_id" class="w-full bg-[#050505] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors" placeholder="e.g. 5052938">
                                    @error('login_id') <span class="text-red-400 text-xs">{{ $message }}</span> @enderror
                                </div>

                                <div>
                                    <label class="block text-sm font-medium text-gray-400 mb-1">Master Password</label>
                                    <input type="password" name="password" class="w-full bg-[#050505] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors" placeholder="••••••••">
                                </div>
                            </div>

                            <div class="pt-4">
                                <button type="submit" class="w-full py-3.5 rounded-xl bg-gradient-to-r from-brand-500 to-brand-600 text-black font-bold shadow-lg shadow-brand-500/20 hover:shadow-brand-500/40 transform hover:-translate-y-0.5 transition-all duration-200">
                                    Connect MetaTrader Account
                                </button>
                            </div>
                        </form>
                    </div>

                    <!-- Deriv Tab -->
                    <div x-show="activeTab === 'deriv'" x-transition:enter="transition ease-out duration-300" x-transition:enter-start="opacity-0 translate-y-2" x-transition:enter-end="opacity-100 translate-y-0" style="display: none;">
                        <form action="{{ route('accounts.connect') }}" method="POST" class="max-w-2xl mx-auto space-y-5">
                            @csrf
                            <input type="hidden" name="type" value="deriv">
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-1">Deriv Server</label>
                                <select name="server" class="w-full bg-[#050505] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors">
                                    <option value="Deriv-Server">Deriv Server</option>
                                    <option value="Deriv-Demo">Deriv Demo</option>
                                </select>
                            </div>
                                
                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-1">Login ID (CR/VR)</label>
                                <input type="text" name="login_id" class="w-full bg-[#050505] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors" placeholder="CR123456">
                            </div>

                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-1">API Token / Password</label>
                                <input type="password" name="password" class="w-full bg-[#050505] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors" placeholder="••••••••">
                            </div>

                            <div class="pt-4">
                                <button type="submit" class="w-full py-3.5 rounded-xl bg-gradient-to-r from-red-500 to-red-600 text-white font-bold shadow-lg shadow-red-500/20 hover:shadow-red-500/40 transform hover:-translate-y-0.5 transition-all duration-200">
                                    Connect Deriv Account
                                </button>
                            </div>
                        </form>
                    </div>

                    <!-- Crypto Tab -->
                    <div x-show="activeTab === 'crypto'" x-transition:enter="transition ease-out duration-300" x-transition:enter-start="opacity-0 translate-y-2" x-transition:enter-end="opacity-100 translate-y-0" style="display: none;">
                        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                             <!-- Exchange Item -->
                             <button class="group p-6 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-brand-500/30 transition-all flex flex-col items-center gap-3 relative overflow-hidden">
                                <div class="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm z-10">
                                    <span class="text-xs font-bold text-white uppercase tracking-wider">Coming Soon</span>
                                </div>
                                <div class="w-12 h-12 rounded-full bg-[#F3BA2F] flex items-center justify-center text-black font-bold text-xl">
                                    <svg class="w-6 h-6" viewBox="0 0 24 24" fill="currentColor"><path d="M16.624 13.9202l2.7175 2.7154-7.353 7.353-7.353-7.352 2.7175-2.7164 4.6355 4.6595 4.6356-4.6595zm4.6366-4.6366L24 12l-2.7154 2.7164L18.5682 12l2.6924-2.7164zm-9.272.331l4.6366-4.6366L12 2.3448l-4.6366 4.633L11.9886 11.9772zM2.7154 9.2836L0 12l2.7154 2.7164 2.7164-2.7164L2.7154 9.2836z"/></svg>
                                </div>
                                <span class="font-bold text-sm text-gray-300">Binance</span>
                             </button>

                             <button class="group p-6 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-brand-500/30 transition-all flex flex-col items-center gap-3 relative overflow-hidden">
                                <div class="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm z-10">
                                    <span class="text-xs font-bold text-white uppercase tracking-wider">Coming Soon</span>
                                </div>
                                <div class="w-12 h-12 rounded-full bg-black border border-white/20 flex items-center justify-center text-white font-bold text-xl">
                                    <svg class="w-6 h-6" viewBox="0 0 24 24" fill="currentColor"><path d="M24 12c0 6.627-5.373 12-12 12S0 18.627 0 12 5.373 0 12 0s12 5.373 12 12z M10.42 16.58h3.16l4.63-8.83L14.7 7H7.72l-4.63 8.83 3.51.75z"/></svg>
                                </div>
                                <span class="font-bold text-sm text-gray-300">Bybit</span>
                             </button>

                             <button class="group p-6 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-brand-500/30 transition-all flex flex-col items-center gap-3 relative overflow-hidden">
                                <div class="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm z-10">
                                    <span class="text-xs font-bold text-white uppercase tracking-wider">Coming Soon</span>
                                </div>
                                <div class="w-12 h-12 rounded-full bg-[#0051C6] flex items-center justify-center text-white font-bold text-xl">C</div>
                                <span class="font-bold text-sm text-gray-300">Coinbase</span>
                             </button>

                             <button class="group p-6 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-brand-500/30 transition-all flex flex-col items-center gap-3 relative overflow-hidden">
                                <div class="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm z-10">
                                    <span class="text-xs font-bold text-white uppercase tracking-wider">Coming Soon</span>
                                </div>
                                <div class="w-12 h-12 rounded-full bg-white text-black flex items-center justify-center font-bold text-xl">O</div>
                                <span class="font-bold text-sm text-gray-300">OKX</span>
                             </button>

                             <button class="group p-6 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-brand-500/30 transition-all flex flex-col items-center gap-3 relative overflow-hidden">
                                <div class="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm z-10">
                                    <span class="text-xs font-bold text-white uppercase tracking-wider">Coming Soon</span>
                                </div>
                                <div class="w-12 h-12 rounded-full bg-[#20B2AA] flex items-center justify-center text-white font-bold text-xl">B</div>
                                <span class="font-bold text-sm text-gray-300">Bitget</span>
                             </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</x-app-layout>
