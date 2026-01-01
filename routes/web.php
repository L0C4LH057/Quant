<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\SubscriptionController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('welcome');
});

Route::middleware(['auth', 'verified'])->group(function () {
    Route::get('/dashboard', [\App\Http\Controllers\DashboardController::class, 'index'])->name('dashboard');
    Route::post('/dashboard/toggle-ai/{id}', [\App\Http\Controllers\DashboardController::class, 'toggleAi'])->name('dashboard.toggle-ai');

    Route::get('/accounts', [\App\Http\Controllers\BrokerController::class, 'index'])->name('accounts.index');

    Route::get('/trading', function () {
        return view('trading.index');
    })->name('trading.index');
    // AI Strategy Routes
    Route::get('/ai-trading', [\App\Http\Controllers\AiStrategyController::class, 'index'])->name('ai-trading.index');
    Route::post('/ai-trading', [\App\Http\Controllers\AiStrategyController::class, 'update'])->name('ai-trading.update');

    // AI Insights Routes
    Route::get('/ai-insights', [\App\Http\Controllers\AiInsightsController::class, 'index'])->name('ai-insights.index');


    Route::get('/settings', function () {
        return view('settings.index');
    })->name('settings.index');

    Route::post('/accounts/connect', [\App\Http\Controllers\BrokerController::class, 'connect'])->name('accounts.connect');
    Route::post('/accounts/{id}/sync', [\App\Http\Controllers\BrokerController::class, 'sync'])->name('accounts.sync');
});

Route::middleware('auth')->group(function () {
    Route::get('/subscription', [SubscriptionController::class, 'index'])->name('subscription.index');
    Route::get('/subscription/manage', [SubscriptionController::class, 'manage'])->name('subscription.manage');
    Route::post('/subscription/upgrade', [SubscriptionController::class, 'upgrade'])->name('subscription.upgrade');
    Route::post('/subscription/cancel', [SubscriptionController::class, 'cancel'])->name('subscription.cancel');
});

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

Route::prefix('admin')->group(function () {
    // Public Admin Routes (Login)
    Route::get('/login', [App\Http\Controllers\AdminAuthController::class, 'create'])->name('admin.login');
    Route::post('/login', [App\Http\Controllers\AdminAuthController::class, 'store'])->name('admin.login.store');
    Route::post('/logout', [App\Http\Controllers\AdminAuthController::class, 'destroy'])->name('admin.logout'); // logout logic

    // Protected Admin Routes
    Route::middleware(['auth', 'admin'])->name('admin.')->group(function () {
        Route::get('/', [App\Http\Controllers\AdminController::class, 'dashboard'])->name('dashboard');
        Route::get('/users', [App\Http\Controllers\AdminController::class, 'users'])->name('users.index');
        Route::get('/trading', [App\Http\Controllers\AdminController::class, 'trading'])->name('trading.index');
    });
});

Route::prefix('api')->group(function () {
    Route::get('/trading/history/{symbol}', [\App\Http\Controllers\Api\TradingDataController::class, 'getHistory'])->name('api.trading.history');
});

require __DIR__.'/auth.php';
