<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('trading_accounts', function (Blueprint $table) {
            $table->decimal('equity', 15, 2)->default(0.00)->after('balance');
            $table->decimal('margin', 15, 2)->default(0.00)->after('equity');
            $table->string('currency')->default('USD')->after('margin');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('trading_accounts', function (Blueprint $table) {
            $table->dropColumn(['equity', 'margin', 'currency']);
        });
    }
};
