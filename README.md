# Spend Data Management Platform - MVP

A comprehensive Streamlit-based application for managing spend data, vendor information, and categorization based on the Low Level Design specifications.

## Features

### 🔐 Authentication & Authorization
- Role-based access control (Admin, Spend Manager, Data Analyst)
- Secure session management
- Demo credentials for testing

### 📊 Dashboard
- Real-time spend analytics
- Key performance indicators (KPIs)
- Interactive charts and visualizations
- Recent transactions overview

### 📤 Data Upload & Validation
- Excel file upload support
- Data validation and error detection
- Preview and confirmation workflow
- Automatic data processing

### 🗂️ Master Data Management
- Vendor management (CRUD operations)
- Category hierarchy management
- Audit trail for changes

### ❌ Error Management
- Error detection and tracking
- Bulk error resolution
- Error reporting and analytics

### 📈 Reports & Analytics
- Interactive filtering (Region, Supplier, Category)
- Trend analysis and visualizations
- Export capabilities (Excel, CSV)
- Custom report generation

## Installation

### Prerequisites
- Python 3.13+
- uv package manager

### Setup

1. **Clone or download the project:**
   ```bash
   cd /path/to/spendplatform
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Run the application:**
   ```bash
   uv run streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

### Running the setup script (macOS / Linux / WSL)

If you're on macOS, Linux, or using WSL on Windows, run the provided shell setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Verify `uv` is available
- Run `uv sync` to install dependencies
- Initialize the database (`init_db.py`)
- Load sample data (`load_sample_data.py`)

If `uv` is not installed, follow the instructions at https://astral.sh/uv to install it.

### Running the setup script (Windows)

Windows users can run the batch equivalent:

1. Open Command Prompt (cmd.exe) as Administrator.
2. From the project root run:

```bat
setup.bat
```

This batch script performs the same steps as the shell script and will notify you if `uv` is not found.

### Starting the application

After setup completes, start the app with:

```bash
uv run streamlit run app.py
```

Then open your browser and go to `http://localhost:8501`.

## Demo Credentials

- **Admin**: `admin` / `admin123`
- **Spend Manager**: `manager` / `manager123`
- **Data Analyst**: `analyst` / `analyst123`

## Data Structure

The application expects spend data with the following key columns:

### Spend Transactions
- `BU_CODE`, `BU _NAME` - Business Unit information
- `Region` - Geographic region
- `PO_NO`, `PO_ITEM_NO` - Purchase Order details
- `Invoice Date` - Transaction date
- `SUPPLIER_NO`, `SUPPLIER_NAME` - Vendor information
- `Item Invoice Value` - Transaction amount
- `CURRENCY_TYPE` - Currency code
- `Material Code`, `Material Item Name` - Product details
- `Tower (Practice)` - Business practice/tower
- `Category`, `Subcategory` - Classification

### Sample Data Files
The `context/` folder contains sample files:
- `sample spend data- filled.xlsx` - Sample spend transactions
- `Categorization File.xlsx` - Category mapping data

## Usage Workflow

### 1. Login
- Use demo credentials to access the platform
- Different roles have different access levels

### 2. Upload Data
- Navigate to "Data Upload" page
- Upload Excel files with spend data
- Review validation results
- Process and save validated data

### 3. Master Data Management
- Manage vendor information
- Create and maintain category hierarchies
- Ensure data consistency

### 4. Error Management
- Review and resolve data errors
- Track error resolution progress
- Generate error reports

### 5. Analytics & Reporting
- View real-time dashboards
- Apply filters for targeted analysis
- Export reports in multiple formats

## Database Schema

The application uses SQLite with the following tables:

- `users` - User authentication and roles
- `vendors` - Vendor master data
- `categories` - Hierarchical category structure
- `spend_transactions` - Main transaction data
- `error_logs` - Error tracking and resolution
- `rules` - Business rules configuration

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: SQLite
- **Visualization**: Plotly
- **Data Processing**: Pandas
- **File Handling**: openpyxl, xlsxwriter

## Development

### Project Structure
```
spendplatform/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── pyproject.toml     # uv project configuration
├── spend_platform.db  # SQLite database (auto-created)
├── context/           # Sample data files
└── README.md          # This file
```

### Adding Features
1. Database schema modifications in `init_database()`
2. New pages as functions (e.g., `new_feature_page()`)
3. Navigation updates in `sidebar_navigation()`
4. Route handling in `main()`

### Data Validation Rules
- Required fields validation
- Negative value detection
- Missing supplier name warnings
- Custom business rule validation

## Production Considerations

For production deployment, consider:

1. **Security**:
   - Implement proper password hashing
   - Use environment variables for secrets
   - Enable HTTPS

2. **Database**:
   - Migrate from SQLite to PostgreSQL/MySQL
   - Implement connection pooling
   - Add data backup strategies

3. **Performance**:
   - Add caching for large datasets
   - Implement pagination for large tables
   - Optimize database queries

4. **Monitoring**:
   - Add logging and monitoring
   - Error tracking and alerting
   - Performance metrics

## License

This is a MVP/demo application for spend data management platform development.
