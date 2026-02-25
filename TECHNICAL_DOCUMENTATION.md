# PipFlow AI - Technical Documentation

## Overview
PipFlow AI is a Laravel-based intelligent trading platform that automates Forex trading using AI-driven strategies. The platform integrates with trading APIs (MetaApi, Deriv) to execute trades, analyze markets, and manage risk automatically.

## System Architecture

### Technology Stack
- **Backend**: Laravel 12 (PHP 8.2+)
- **Frontend**: Blade templates with Tailwind CSS
- **Database**: MySQL/SQLite (via Eloquent ORM)
- **Queue**: Laravel Queue (for async trading operations)
- **Authentication**: Laravel Breeze with Socialite (Google OAuth)
- **API Integration**: MetaApi, Deriv Trading APIs

### Directory Structure
```
Quant/
├── app/
│   ├── Http/Controllers/          # Application controllers
│   ├── Models/                    # Database models
│   ├── Services/Trading/          # Trading API services
│   ├── Providers/                 # Service providers
│   └── View/                      # View components
├── config/                        # Configuration files
├── database/
│   ├── migrations/               # Database migrations
│   └── seeders/                  # Database seeders
├── resources/
│   ├── views/                    # Blade templates
│   ├── js/                       # JavaScript files
│   └── css/                      # CSS files
├── routes/                       # Application routes
└── public/                       # Public assets
```

## Database Schema

### Core Tables

#### 1. users
```sql
- id (PK)
- name
- email
- email_verified_at
- password
- google_id (nullable)
- subscription_plan (enum: free, basic, pro, enterprise)
- role (enum: user, admin)
- remember_token
- created_at
- updated_at
```

#### 2. trading_accounts
```sql
- id (PK)
- user_id (FK to users)
- name
- broker_type (enum: metaapi, deriv)
- account_id
- meta_api_id (nullable)
- is_ai_active (boolean)
- equity (decimal)
- created_at
- updated_at
```

#### 3. ai_configurations
```sql
- id (PK)
- user_id (FK to users)
- trading_account_id (FK to trading_accounts)
- strategy_type
- risk_level
- trading_pairs
- stop_loss_percentage
- take_profit_percentage
- max_daily_trades
- created_at
- updated_at
```

#### 4. Other Tables
- `cache` - Laravel cache table
- `jobs` - Queue jobs table
- `sessions` - User sessions

## API Integrations

### 1. MetaApi Service
**Location**: `app/Services/Trading/MetaApiService.php`
- Connects to MetaApi trading platform
- Fetches account information
- Retrieves trading history
- Executes trades
- Manages positions

### 2. Deriv Service
**Location**: `app/Services/Trading/DerivService.php`
- Connects to Deriv trading API
- Handles binary options trading
- Manages contracts
- Processes payouts

### 3. Trading Service Factory
**Location**: `app/Services/Trading/TradingServiceFactory.php`
- Factory pattern for trading services
- Returns appropriate service based on broker type
- Implements `TradingServiceInterface`

## Code Structure

### Models
1. **User** (`app/Models/User.php`)
   - Manages user authentication and profiles
   - Handles subscription plans
   - Relationships: hasMany TradingAccount, hasOne AiConfiguration

2. **TradingAccount** (`app/Models/TradingAccount.php`)
   - Represents connected trading accounts
   - Stores broker credentials and settings
   - Relationships: belongsTo User, hasOne AiConfiguration

3. **AiConfiguration** (`app/Models/AiConfiguration.php`)
   - Stores AI trading strategy settings
   - Configures risk management parameters
   - Relationships: belongsTo User, belongsTo TradingAccount

### Controllers
1. **DashboardController** - Main user dashboard
2. **BrokerController** - Trading account management
3. **AiStrategyController** - AI configuration management
4. **AiInsightsController** - Trading insights and analytics
5. **SubscriptionController** - Subscription management
6. **AdminController** - Admin panel operations
7. **Api\TradingDataController** - API endpoints for trading data

### Services
1. **TradingServiceInterface** - Interface for trading services
2. **MetaApiService** - MetaApi implementation
3. **DerivService** - Deriv implementation
4. **TradingServiceFactory** - Service factory

## Configuration Guide

### Environment Variables
```env
APP_NAME=PipFlow AI
APP_ENV=production
APP_KEY=
APP_DEBUG=false
APP_URL=http://localhost

DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=pipflow
DB_USERNAME=root
DB_PASSWORD=

META_API_TOKEN=your_metaapi_token
DERIV_APP_ID=your_deriv_app_id

GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost/auth/google/callback

STRIPE_KEY=your_stripe_key
STRIPE_SECRET=your_stripe_secret
```

### Service Configuration
1. **MetaApi Setup**:
   - Obtain API token from MetaApi
   - Configure in `.env` as `META_API_TOKEN`
   - Test connection using `debug_meta_api.php`

2. **Deriv Setup**:
   - Register app on Deriv
   - Get App ID and configure in `.env`
   - Test connection using trading endpoints

3. **Google OAuth**:
   - Create project in Google Cloud Console
   - Configure OAuth 2.0 credentials
   - Set redirect URI

## Deployment Instructions

### Prerequisites
- PHP 8.2+
- Composer
- Node.js 18+
- MySQL 5.7+ or SQLite
- Web server (Nginx/Apache)

### Installation Steps
1. Clone repository:
   ```bash
   git clone <repository-url>
   cd Quant
   ```

2. Install dependencies:
   ```bash
   composer install
   npm install
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   php artisan key:generate
   ```

4. Update `.env` with your configuration

5. Run migrations:
   ```bash
   php artisan migrate
   ```

6. Build assets:
   ```bash
   npm run build
   ```

7. Set up queue worker (for async trading):
   ```bash
   php artisan queue:work
   ```

8. Configure web server to point to `public/` directory

### Development Setup
```bash
# Run development server
php artisan serve

# Run Vite for frontend
npm run dev

# Run queue worker
php artisan queue:listen

# Run tests
php artisan test
```

## Security Considerations

### 1. API Security
- Trading API tokens stored encrypted
- Rate limiting on trading endpoints
- IP whitelisting for sensitive operations

### 2. User Data Protection
- Passwords hashed with bcrypt
- Session management with secure cookies
- CSRF protection on all forms

### 3. Trading Security
- Risk limits per user/account
- Maximum drawdown protection
- Trade size limits based on equity

### 4. Financial Security
- No direct access to user funds
- Trading through regulated brokers only
- Audit logs for all trading activities

## Monitoring & Maintenance

### Logging
- Laravel logging to `storage/logs/`
- Trading activity logs
- Error tracking with Sentry (optional)

### Performance Monitoring
- Queue monitoring for trading jobs
- Database query optimization
- API response time tracking

### Backup Strategy
- Daily database backups
- Configuration file versioning
- Disaster recovery plan

## Troubleshooting

### Common Issues
1. **API Connection Failures**
   - Check API tokens in `.env`
   - Verify network connectivity
   - Check broker account status

2. **Queue Jobs Not Processing**
   - Ensure queue worker is running
   - Check Redis/queue connection
   - Monitor failed jobs table

3. **Database Migration Issues**
   - Clear migration cache: `php artisan migrate:refresh`
   - Check database permissions
   - Verify .env database configuration

### Debug Tools
- `debug_meta_api.php` - MetaApi connection test
- `debug_history_filter.php` - Trading history debug
- `manual_test.php` - Manual trading tests
- Laravel Telescope (optional for dev)

## Scaling Considerations

### Horizontal Scaling
- Database replication for read-heavy operations
- Multiple queue workers for trading jobs
- Load balancing for web servers

### Vertical Scaling
- Database optimization with indexes
- Caching frequently accessed data
- CDN for static assets

### Cost Optimization
- Queue batching for trading operations
- Database query optimization
- Caching strategy implementation

---

*Last Updated: January 8, 2026*
*Version: 1.0.0*
