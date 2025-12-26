import pandas as pd
import numpy as np
from collections import Counter

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)
# ---- FILE PATHS (edit if your files are in another folder) ----
ORDERS_PATH   = "orders.csv"
PRIOR_PATH    = "order_products_prior.csv"

PRODUCTS_PATH = "products.csv"
AISLES_PATH   = "aisles.csv"
DEPTS_PATH    = "departments.csv"
# ---- Load lookup tables (small) ----
products = pd.read_csv(
    PRODUCTS_PATH,
    usecols=["product_id", "product_name", "aisle_id", "department_id"],
    dtype={"product_id":"int32", "aisle_id":"int16", "department_id":"int16"}
)

aisles = pd.read_csv(AISLES_PATH, dtype={"aisle_id":"int16"})
departments = pd.read_csv(DEPTS_PATH, dtype={"department_id":"int16"})

print("Loaded products/aisles/departments")
print(products.shape, aisles.shape, departments.shape)
# ---- Load orders with only needed columns ----
orders = pd.read_csv(
    ORDERS_PATH,
    usecols=["order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"],
    dtype={
        "order_id":"int32",
        "user_id":"int32",
        "eval_set":"category",
        "order_number":"int16",
        "order_dow":"int8",
        "order_hour_of_day":"int8",
        "days_since_prior_order":"float32"
    }
)

# ---- Clean: remove nulls in critical columns ----
orders = orders.dropna(subset=["order_id", "user_id", "eval_set"])

print("orders:", orders.shape)
orders.head()
# ---- SUBSET: sample 20,000 users to keep dataset manageable ----
N_USERS = 20_000
rng = np.random.default_rng(42)

unique_users = orders["user_id"].unique()
sample_users = rng.choice(unique_users, size=min(N_USERS, len(unique_users)), replace=False)

orders_sub = orders[orders["user_id"].isin(sample_users)].copy()

# We focus on 'prior' orders for Market Basket Analysis (baskets)
orders_sub = orders_sub[orders_sub["eval_set"].isin(["prior"])].copy()

eligible_order_ids = set(orders_sub["order_id"].tolist())

print("orders_sub:", orders_sub.shape)
print("eligible_order_ids:", len(eligible_order_ids))
# ---- Chunk-read order_products_prior and keep only sampled orders ----
op_cols = ["order_id", "product_id"]
op_dtypes = {"order_id":"int32", "product_id":"int32"}

CHUNKSIZE = 2_000_000   # if RAM is low, use 500_000
kept_chunks = []
order_item_counts = Counter()

for chunk in pd.read_csv(PRIOR_PATH, usecols=op_cols, dtype=op_dtypes, chunksize=CHUNKSIZE):
    # remove nulls in critical fields
    chunk = chunk.dropna(subset=["order_id", "product_id"])

    # keep only sampled order_ids
    chunk = chunk[chunk["order_id"].isin(eligible_order_ids)]

    # remove duplicates (same product repeated in same order)
    chunk = chunk.drop_duplicates(subset=["order_id", "product_id"])

    # count items per order (for removing 1-item baskets)
    order_item_counts.update(chunk["order_id"].tolist())

    kept_chunks.append(chunk)

order_products_sub = pd.concat(kept_chunks, ignore_index=True)

print("order_products_sub:", order_products_sub.shape)
order_products_sub.head()
# ---- Remove orders with only 1 item (basket must have >=2 items) ----
valid_orders = {oid for oid, c in order_item_counts.items() if c >= 2}

order_products_sub = order_products_sub[order_products_sub["order_id"].isin(valid_orders)].copy()
orders_sub = orders_sub[orders_sub["order_id"].isin(valid_orders)].copy()

print("After removing 1-item orders:")
print("orders_sub:", orders_sub.shape)
print("order_products_sub:", order_products_sub.shape)
# ---- OPTIONAL: remove very rare products to reduce columns later ----
MIN_PRODUCT_COUNT = 50   # try 20, 50, 100 (higher = faster, fewer rules)

prod_counts = order_products_sub["product_id"].value_counts()
keep_products = set(prod_counts[prod_counts >= MIN_PRODUCT_COUNT].index.astype("int32"))

order_products_sub = order_products_sub[order_products_sub["product_id"].isin(keep_products)].copy()

print("After removing rare products:")
print("order_products_sub:", order_products_sub.shape)
print("Unique products kept:", order_products_sub["product_id"].nunique())
# ---- Add product names (merge after filtering) ----
order_products_sub = order_products_sub.merge(
    products[["product_id", "product_name"]],
    on="product_id",
    how="left"
)

# remove any rows that failed the merge (should be rare)
order_products_sub = order_products_sub.dropna(subset=["product_name"])

print("Final cleaned order_products_sub:", order_products_sub.shape)
order_products_sub.head()
print("Transactions (orders):", orders_sub["order_id"].nunique())
print("Unique products:", order_products_sub["product_id"].nunique())
print("Lines (order-product rows):", len(order_products_sub))
