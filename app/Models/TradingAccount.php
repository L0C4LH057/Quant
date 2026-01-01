<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class TradingAccount extends Model
{
    protected $fillable = [
        'user_id',
        'name',
        'broker_type',
        'login_id',
        'server',
        'password',
        'status',
        'balance',
        'equity',
        'margin',
        'currency',
        'meta_api_id',
        'is_ai_active',
    ];

    protected $casts = [
        'is_ai_active' => 'boolean',
        'balance' => 'decimal:2',
        'equity' => 'decimal:2',
        'margin' => 'decimal:2',
        'password' => 'encrypted',
    ];

    // Accessor to allow $account->login to work (aliases login_id)
    public function getLoginAttribute()
    {
        return $this->attributes['login_id'];
    }

    // Accessor to allow $account->type to work (aliases broker_type)
    public function getTypeAttribute()
    {
        return $this->attributes['broker_type'];
    }
}
