import pandas as pd
import numpy as np
import os
import re
from thefuzz import process
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import yaml
import importlib.resources

# SET UP ENVIRONMENT AND MODEL
dotenv.load_dotenv()
model = OpenAIServerModel(
    model_id="gpt-3.5-turbo", #gpt-4o-mini
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
@tool
def check_stock_levels(item_name: str, as_of_date: str) -> str:
    """
    Checks the stock level for a given item.

    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date to check the stock level on, in YYYY-MM-DD format.

    Returns:
        A string representation of the DataFrame containing the stock level.
    """
    stock_level = get_stock_level(item_name, as_of_date)
    return stock_level.to_string()

@tool
def check_reorder_status(item_name: str, as_of_date: str) -> str:
    """
    Checks if an item needs to be reordered by comparing the current stock to the minimum stock level.

    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date to check the reorder status on, in YYYY-MM-DD format.

    Returns:
        A string indicating whether the item needs to be reordered.
    """
    stock_level_df = get_stock_level(item_name, as_of_date)
    if stock_level_df.empty or stock_level_df.iloc[0]["current_stock"] is None:
        return f"Could not determine stock level for {item_name}."

    current_stock = stock_level_df.iloc[0]["current_stock"]

    inventory_df = pd.read_sql("SELECT min_stock_level FROM inventory WHERE item_name = :item_name", db_engine, params={"item_name": item_name})

    if inventory_df.empty:
        return f"Item {item_name} not found in inventory."

    min_stock_level = inventory_df.iloc[0]["min_stock_level"]

    if current_stock < min_stock_level:
        return f"Item {item_name} needs to be reordered. Current stock: {current_stock}, Minimum stock: {min_stock_level}."
    else:
        return f"Item {item_name} is sufficiently stocked. Current stock: {current_stock}, Minimum stock: {min_stock_level}."

@tool
def place_stock_order(item_name: str, quantity: int, price: float, date: str) -> str:
    """
    Places a stock order for a given item.

    Args:
        item_name (str): The name of the item to order.
        quantity (int): The number of units to order.
        price (float): The total price of the order.
        date (str): The date of the order, in YYYY-MM-DD format.

    Returns:
        A string confirming the stock order and providing a transaction ID, or an error message.
    """
    try:
        transaction_id = create_transaction(item_name, "stock_orders", quantity, price, date)
        return f"Stock order placed successfully. Transaction ID: {transaction_id}"
    except Exception as e:
        return f"Error placing stock order: {e}"

@tool
def get_full_inventory_report(as_of_date: str) -> str:
    """
    Provides a full report of all items in inventory with their current stock levels.

    Args:
        as_of_date (str): The date for the report, in YYYY-MM-DD format.

    Returns:
        A string representation of the inventory report.
    """
    inventory_dict = get_all_inventory(as_of_date)
    if not inventory_dict:
        return "No inventory found."
    return pd.DataFrame.from_dict(inventory_dict, orient='index', columns=['stock']).to_string()

@tool
def check_cash_balance(as_of_date: str) -> str:
    """
    Checks the current cash balance of the company.

    Args:
        as_of_date (str): The date to check the balance on, in YYYY-MM-DD format.

    Returns:
        A string stating the current cash balance.
    """
    balance = get_cash_balance(as_of_date)
    return f"The current cash balance is ${balance:.2f}."

@tool
def get_company_financials(as_of_date: str) -> str:
    """
    Generates a complete financial report for the company for internal use.

    Args:
        as_of_date (str): The date for the report, in YYYY-MM-DD format.

    Returns:
        A string summarizing the financial report.
    """
    report = generate_financial_report(as_of_date)
    return (
        f"Financial Report as of {report['as_of_date']}:\n"
        f"Cash Balance: ${report['cash_balance']:.2f}\n"
        f"Inventory Value: ${report['inventory_value']:.2f}\n"
        f"Total Assets: ${report['total_assets']:.2f}\n"
        f"Top Selling Products: {report['top_selling_products']}"
    )


# Tools for quoting agent
@tool
def quote_history(customer_request: str) -> str:
    """
    Searches for past quotes based on a customer's request.

    Args:
        customer_request (str): The customer's request to search for in the quote history.

    Returns:
        A string representation of the DataFrame containing past quotes, or a message if none are found.
    """
    try:
        # Use the helper function by splitting the request into search terms
        search_terms = customer_request.split()
        quotes = search_quote_history(search_terms) # The helper function is called here
        if not quotes:
            return "No similar past quotes found."
        return pd.DataFrame(quotes).to_string()
    except Exception as e:
        return f"Error searching quote history: {e}"


@tool
def get_pricing_and_availability(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Gets the current price, availability, and estimated delivery date for a given item and quantity.
    Applies a bulk discount for larger orders.

    Args:
        item_name (str): The name of the item to check.
        quantity (int): The number of units being requested.
        as_of_date (str): The date to check the price and availability on, in YYYY-MM-DD format.

    Returns:
        A string with the item's price, availability, and estimated delivery date, or an error message.
    """
    try:
        inventory_df = pd.read_sql("SELECT unit_price FROM inventory WHERE item_name = :item_name", db_engine,
                                   params={"item_name": item_name})
        if inventory_df.empty:
            return f"Item {item_name} not found in inventory."

        unit_price = inventory_df.iloc[0]["unit_price"]
        stock_level_df = get_stock_level(item_name, as_of_date)
        current_stock = stock_level_df.iloc[0]['current_stock']

        total_price = unit_price * quantity

        delivery_date = get_supplier_delivery_date(as_of_date, quantity)

        return (f"Item: {item_name}, Price per unit: ${unit_price:.2f}, "
                f"Total for {quantity} units: ${total_price:.2f}. "
                f"Current Availability: {current_stock} units. "
                f"Estimated delivery date: {delivery_date}.")
    except Exception as e:
        return f"Error getting pricing and availability: {e}"


@tool
def apply_commission_and_discount(base_quote_str: str, discount_rate: float) -> str:
    """
    Applies a standard 5% sales commission and a variable loyalty discount to a base quote.
    The agent determines the discount rate based on quote history.

    Args:
        base_quote_str (str): The string output from the `get_pricing_and_availability` tool.
        discount_rate (float): The loyalty discount rate to apply, as a decimal (e.g., 0.02 for 2%).

    Returns:
        A final quote string including commission and the specified discount.
    """
    try:
        # Extract total price and quantity from the base quote string
        price_match = re.search(r"Total for (\d+) units: \$([\d\.]+)\.", base_quote_str)
        if not price_match:
            return "Error: Could not parse base price from the quote string."

        quantity = int(price_match.group(1))
        base_price = float(price_match.group(2))

        # Apply a standard 5% sales commission
        sale_commission = 1.05
        price_after_commission = base_price * sale_commission

        # Apply the variable discount decided by the agent
        discount_amount = base_price * discount_rate
        final_price = price_after_commission - discount_amount

        discount_explanation = ""
        if discount_rate > 0:
            discount_percentage = discount_rate * 100
            discount_explanation += (
                f" A {discount_percentage:.1f}% discount was also applied for your order, "
                f"saving you an additional ${discount_amount:.2f}.")

        # Replace the original total price with the new final price in the quote string
        final_quote_str = re.sub(
            r"Total for \d+ units: \$[\d\.]+\.",
            f"Total for {quantity} units: ${final_price:.2f}. {discount_explanation}",
            base_quote_str
        )

        return final_quote_str
    except Exception as e:
        return f"Error applying commission and discount: {e}"

# Tools for ordering agent
@tool
def finalize_order(item_name: str, quantity: int, price: float, date: str) -> str:
    """
    Finalizes a customer's order by creating a 'sales' transaction.

    Args:
        item_name (str): The name of the item being ordered.
        quantity (int): The number of units being ordered.
        price (float): The total price of the order.
        date (str): The date of the order, in YYYY-MM-DD format.

    Returns:
        A string confirming the order and providing a transaction ID, or an error message.
    """
    try:
        stock_level_df = get_stock_level(item_name, date)
        if stock_level_df.empty or stock_level_df.iloc[0]["current_stock"] < quantity:
            return f"Insufficient stock for {item_name}. Order cannot be finalized."

        transaction_id = create_transaction(item_name, "sales", quantity, price, date)
        return f"Order finalized successfully. Transaction ID: {transaction_id}"
    except Exception as e:
        return f"Error finalizing order: {e}"

@tool
def normalize_item_names(user_request: str, as_of_date: str) -> str:
    """
    Finds phrases in a user request that describe items, normalizes them against official
    inventory names, and returns the request with the names replaced.
    This preserves the original structure and context of the user's request.

    Args:
        user_request (str): The original user request string.
        as_of_date (str): The date for checking inventory, in YYYY-MM-DD format.

    Returns:
        The user request string with item names normalized.
    """
    inventory_item_names = list(get_all_inventory(as_of_date).keys())
    if not inventory_item_names:
        return user_request

    def find_and_replace(match):
        # The part of the string that contains the quantity and units
        quantity_and_units = match.group(1)
        # The part of the string that is the item description
        original_item_phrase = match.group(2).strip()

        # Clean the phrase to improve matching accuracy by removing common, non-specific words
        cleaned_phrase = re.sub(r'\([\w\s-]+\)', '', original_item_phrase, flags=re.IGNORECASE).strip()
        cleaned_phrase = re.sub(r'\b(high-quality|white|assorted colors|various colors|standard)\b', '', cleaned_phrase, flags=re.IGNORECASE).strip()

        if not cleaned_phrase:
            return match.group(0) # Return original full match if cleaning results in empty string

        # Find the best match in the inventory for the cleaned phrase
        best_match, score = process.extractOne(cleaned_phrase, inventory_item_names)

        # If confidence is high, reconstruct the string with the normalized item name
        if score > 80:
            return f"{quantity_and_units}{best_match}"
        else:
            # Otherwise, return the original matched text without changes
            return match.group(0)

    # This regex captures two groups:
    # 1. The quantity and any common units (e.g., "200 sheets of ").
    # 2. The descriptive item phrase that follows (e.g., "A4 glossy paper").
    # It stops capturing before a newline, comma, or the word "and".
    # The units part now handles both singular and plural forms (e.g., sheet/sheets).
    pattern = re.compile(
        r'(\d[\d,]*\s+(?:sheets? of|rolls? of|reams? of|packets? of|of|)?\s*)([a-zA-Z0-9\s\(\)\'\"-]+?)(?=\n|,|\s+and\s+|\.|$)',
        re.IGNORECASE
    )

    return pattern.sub(find_and_replace, user_request)


# Set up your agents and create an orchestration agent that will manage them.
INVENTORY_SYSTEM_PROMPT = """You are a specialized agent for managing inventory. Your goal is to proactively maintain stock levels and respond to specific inventory-related tasks by following a strict workflow and rules.

Here are your available tools:
- `check_stock_levels`: For checking a specific item's stock.
- `check_reorder_status`: For checking if an item's stock is below the minimum threshold.
- `place_stock_order`: For ordering more of a specific item. **This tool has strict usage rules.**
- `check_cash_balance`: For checking the company's cash balance before placing an order.
- `get_full_inventory_report`: For a report of all items in stock.
- `get_company_financials`: For a full financial report containing item unit prices.

**Your Primary Workflow:**

You MUST follow this workflow for EVERY task you receive:

1.  **Proactive Stock Maintenance (Run First)**:
    a.  Use `get_full_inventory_report` to get a list of all items currently in stock.
    b.  For each item in the report, use `check_reorder_status` to identify any items with stock below the minimum required level.
    c.  If any item needs to be reordered, you MUST determine the reorder quantity and then follow the **`place_stock_order` Usage Rule** below to attempt a reorder. The quantity to reorder is `(min_stock_level - current_stock) + 50`.

2.  **Execute Assigned Task**: After completing the proactive stock maintenance, proceed to execute the specific task you were originally assigned (e.g., checking stock, placing a specific order). You MUST adhere to all tool usage rules.

**Tool Usage Rules:**

- **`place_stock_order` Usage Rule**: You MUST follow these steps in sequence BEFORE every single call to `place_stock_order`:
    1.  **Determine Order Cost**: You must know the `item_name`, `quantity`, and `unit_price`.
        - If the task does not provide the `unit_price`, use `get_company_financials` to find it.
        - Calculate the total order cost: `quantity * unit_price`.
    2.  **Check Funds**: Use `check_cash_balance` to get the current cash balance.
    3.  **Verify and Execute**: Compare the total order cost to the cash balance.
        - **If cash balance is greater than or equal to the order cost**, you are authorized to call `place_stock_order`.
        - **If cash balance is insufficient**, you MUST NOT call `place_stock_order`. You must report that the order failed due to insufficient funds.

**Handle Ambiguity**: If a request is ambiguous or you cannot fulfill it, you MUST respond with a summary of your capabilities and ask for clarification. Do not place any orders if the request is ambiguous.
"""

class InventoryAgent(ToolCallingAgent):
    """
    Manages all inventory-related tasks, including checking stock levels, assessing reorder needs,
    and executing the stock reordering workflow.
    """
    def __init__(self, model: OpenAIServerModel):
        # Load default prompt templates
        prompt_templates = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        # Set the custom system prompt
        prompt_templates["system_prompt"] = INVENTORY_SYSTEM_PROMPT

        super().__init__(
            tools=[
                check_stock_levels,
                check_reorder_status,
                place_stock_order,
                get_full_inventory_report,
                check_cash_balance,
                get_company_financials
            ],
            model=model,
            name="inventory_agent",
            prompt_templates=prompt_templates,
            description=(
                "Agent for managing inventory. Handles inquiries about stock levels and inventory reports. "
                "It is also responsible for the reordering workflow: "
                "1. Check if an item needs reordering. "
                "2. If it does, check the company's cash balance to ensure sufficient funds. "
                "3. If there is enough cash, place a stock order to replenish the inventory."
            ),
        )

QUOTING_SYSTEM_PROMPT = """You are a specialized agent for customer quotes. Your goal is to generate a final, consolidated quote for all items in a customer's request by following a strict, multi-step process.

Here are your available tools:
- `get_pricing_and_availability`: For getting the base price, availability, and delivery date for a single item.
- `quote_history`: For searching past quotes based on a customer's request to inform discount decisions.
- `apply_commission_and_discount`: For applying a sales commission and a variable loyalty discount to a single item's base quote.

**Your Primary Workflow:**

You MUST follow this workflow for EVERY task you receive. If the request contains multiple items, you MUST repeat the following steps for EACH item before providing a final, consolidated answer.

1.  **Process Each Item Individually**: For each item in the request:
    a.  **Get Base Quote**: Call `get_pricing_and_availability` to get the fundamental details for the item and its requested quantity.
    b.  **Check History**: Call `quote_history` using the customer's request to see if there are any relevant past orders or discounts that might apply.
    c.  **Analyze and Decide Discount**: Review the output from `quote_history`. Based on the customer's past orders (e.g., large quantities, frequent orders, or previously applied discounts), decide on a reasonable loyalty discount rate for the current item. The rate MUST be a decimal between 0.0 (no discount) and 0.03 (max 3% discount). If no relevant history is found, you MUST use a discount rate of 0.0.
    d.  **Generate Item Quote**: Call `apply_commission_and_discount` with the base quote string from step 'a' and the discount rate you decided on in step 'c'. This will generate the complete quote for this single item.
2.  **Consolidate and Finalize**: After processing all items, combine the individual quotes into a single, clear, and itemized response for the user.

If a request is ambiguous or you cannot fulfill it, you MUST respond with a summary of your capabilities and ask for clarification. Do not provide a quote if the request is ambiguous.
"""

class QuotingAgent(ToolCallingAgent):
    """
    Handles all customer-facing quoting tasks. It provides pricing, checks item availability,
    and searches historical quote data to inform its responses.
    """
    def __init__(self, model: OpenAIServerModel):
        # Load default prompt templates
        prompt_templates = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        # Set the custom system prompt
        prompt_templates["system_prompt"] = QUOTING_SYSTEM_PROMPT

        super().__init__(
            tools=[get_pricing_and_availability, quote_history, apply_commission_and_discount],
            model=model,
            name="quoting_agent",
            description=(
                "Generates final customer quotes. It gets base pricing, checks historical data, "
                "decides on a reasonable loyalty discount, and applies a standard sales commission."
            ),
            prompt_templates=prompt_templates,
        )

ORDERING_SYSTEM_PROMPT = """You are a specialized agent for finalizing customer orders. Your goal is to respond to tasks by calling the correct tool.

Here is your available tool:
- `finalize_order`: For creating a sales transaction for a specific item, quantity, and price.

If a request is ambiguous or you cannot fulfill it, you MUST respond with a summary of the system at your end or best answer to your understanding after clearly mentioning that request was ambigious and you cannot fulfill it and your capabilities and ask for clarification. DONOT execute order if request is ambigious.
"""

class OrderingAgent(ToolCallingAgent):
    """
    Responsible for finalizing sales. It validates stock availability before creating
    a sales transaction in the database to complete a customer's purchase.
    """
    def __init__(self, model: OpenAIServerModel):
        # Load default prompt templates
        prompt_templates = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        # Set the custom system prompt
        prompt_templates["system_prompt"] = ORDERING_SYSTEM_PROMPT

        super().__init__(
            tools=[finalize_order],
            model=model,
            name="ordering_agent",
            description="Finalizes customer orders. It confirms stock availability and then creates a sales transaction to complete the purchase.",
            prompt_templates=prompt_templates,
        )

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Manager acting as an expert and delegator. Your goal is to fulfill user requests by executing an efficient, step-by-step workflow.

Here are the available agents and their responsibilities:
- **inventory_agent**: Use for inquiries about stock levels and for executing the internal reordering workflow if stock is insufficient for a customer order.
- **quoting_agent**: Use for requests for quotes, pricing, and checking product availability.
- **ordering_agent**: Use to finalize a customer order by creating a sales transaction.

You also have access to the following tool:
- `normalize_item_names`: Use this tool first to clean up the user's request by matching item names to the official inventory and replacing them in the request string.

 **Execution Rules**: Execute your plan one step at a time by calling the appropriate agent. Always pass these 2 arguments `task` and `additional_args`. **STRICTLY ONLY** pass these 2 arguments and nothing else.
- `task` is a detailed string that describes the specific goal for the agent. You MUST include all relevant context, especially the date for any date-sensitive tasks and task_outcome.
- `additional_args` MUST be an empty dictionary: `{}` if no additional_args are available.

Here is your `step-by-step` workflow:
**Execution Workflow for Orders**: For any request that involves placing an order, you MUST follow this specific sequence while following `Execution Rules` given earlier:
    1.  **Normalize Request**: Call the `normalize_item_names` tool with the original user request and the request date. This will return a new version of the request with all item names corrected to match the official inventory. Use this normalized request for all subsequent steps.
    2.  **Get Quote**: Call the `quoting_agent` to get a quote for each item in the normalized request. The quote will include the current stock level and an estimated delivery date.
    3.  If current stock is sufficient or if the estimated delivery date matches the user request. User request can be fulfilled.
        - **If current stock is sufficient**: Your next step is to call the `ordering_agent` to `finalize_order`.
        - **Otherwise If current stock is insufficient**: Your next step is to call the `inventory_agent` to reorder the necessary stock. The task should specify the item, the quantity needed, and the user's timeline.
        **Follow-up After Stock Order Attempt**:
        - If the `inventory_agent` reports that the reorder was successful, your next step is to call the `ordering_agent` to `finalize_order`. 
        - If the `inventory_agent` reports that the reorder failed due to valid reason (e.g., due to insufficient funds), the item CANNOT be fulfilled. You MUST NOT proceed with the order. You MUST retry the reorder up to 2 more times before marking it as failed. 
    4.  If current stock is insufficient and the estimated delivery date does not matches the user request either. User request can not be fulfilled.
        - simply send the quote you have from `quoting_agent` to the user.
    **Finalize**: When the plan is complete, you MUST synthesize the final answer for the user.
    - If the order was successful, confirm it.
    - If the order could not be fulfilled, explain why (e.g., "the item is out of stock and could not be reordered in time") and provide the quote for the quantity that is available.
    - DO NOT include tool names, function calls, or raw data logs in your final answer.
"""

class OrchestratorAgent(ToolCallingAgent):
    """
    Acts as an expert planner and delegator. It analyzes user requests, creates an efficient,
    step-by-step plan, and delegates tasks to specialized worker agents. It dynamically reviews
    and adapts the plan based on intermediate results to ensure optimal workflow.
    """
    def __init__(self, model):
        # Instantiate the specialized agents
        inventory_agent = InventoryAgent(model=model)
        quoting_agent = QuotingAgent(model=model)
        ordering_agent = OrderingAgent(model=model)

        # Load default prompt templates for the ToolCallingAgent
        prompt_templates = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        # Set the custom system prompt for the Orchestrator
        prompt_templates["system_prompt"] = ORCHESTRATOR_SYSTEM_PROMPT

        # Initialize the parent ToolCallingAgent
        super().__init__(
            model=model,
            tools=[normalize_item_names],
            # Pass the specialized agents as managed_agents for proper delegation
            managed_agents=[
                inventory_agent,
                quoting_agent,
                ordering_agent,
            ],
            # Pass the customized prompt templates
            prompt_templates=prompt_templates,
            max_steps=10
        )


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    # Sort by date
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"]
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    orchestrator = OrchestratorAgent(model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############
        response = orchestrator.run(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()