# -----------------------------
# MEDIA MIX MODEL WITH FIXED EFFECTS
# -----------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# ---- 1. Load your CSV data ----
df = pd.read_csv("Data.csv")

# ---- 2. Ensure categorical variables ----
df['state'] = df['state'].astype('category')
df['year'] = df['year'].astype('category')
df['brand_name'] = df['brand_name'].astype('category')

# ---- 3. Create dummies for brand, state, year ----
brand_dummies = pd.get_dummies(df['brand_name'], prefix='brand', drop_first=True).astype(float)
state_dummies = pd.get_dummies(df['state'], prefix='state', drop_first=True).astype(float)
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True).astype(float)

# ---- 4. Define base numeric variables ----
base_vars = ['tv_spend', 'radio_spend', 'social_spend', 'price']
X_base = df[base_vars].astype(float)
y = pd.to_numeric(df['sales'], errors='coerce')

# ---- 5. Handle missing/infs ----
X_base.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)

valid_idx = X_base.notna().all(axis=1) & y.notna()
X_base = X_base.loc[valid_idx]
y = y.loc[valid_idx]
brand_dummies = brand_dummies.loc[valid_idx]
state_dummies = state_dummies.loc[valid_idx]
year_dummies = year_dummies.loc[valid_idx]

# ---- 6. Function to calculate Elasticities and ROI ----
def calculate_elasticity_roi(model, X, y):
    """
    Calculates elasticities and ROI for media variables (TV, Radio, Social)
    Elasticity = beta * (mean spend / mean sales)
    ROI = beta (sales per $ spent)
    """
    mean_sales = y.mean()
    results = []
    for var in ['tv_spend', 'radio_spend', 'social_spend', 'price']:
        if var in model.params.index:
            beta = model.params[var]
            mean_var = X[var].mean()
            elasticity = beta * (mean_var / mean_sales)  # % change in sales per % change in spend
            roi = beta  # $ sales generated per $ spend
            results.append({'Variable': var, 'Coefficient': beta, 
                            'Elasticity': elasticity, 'ROI': roi})
    return pd.DataFrame(results)

# ---- 7. Function to run OLS regression ----
def run_ols(X, y, name):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    results_df = calculate_elasticity_roi(model, X, y)
    results_df.to_csv(f'ols_results_{name}.csv', index=False)
    return model, results_df

# ---- 8. Function to run IV regression ----
# Assuming you have instruments: 'avg_electric_price', 'avg_gas_price'
instruments = ['avg_electric_price', 'avg_gas_price']  # Replace with your actual columns
def run_iv(y, X, endog_var, instr, name):
    # Add constant
    X = sm.add_constant(X)
    iv_model = IV2SLS(y, X.drop(columns=[endog_var]), X[endog_var], df[instr]).fit()
    results_df = calculate_elasticity_roi(iv_model, X, y)
    results_df.to_csv(f'iv_results_{name}.csv', index=False)
    return iv_model, results_df

# ---- 9. Define models ----
# Model 1: Base + Brand FE
X1 = pd.concat([X_base, brand_dummies], axis=1)
ols1, df1 = run_ols(X1, y, 'model1_base_brand')

# Model 2: Model 1 + State FE
X2 = pd.concat([X1, state_dummies], axis=1)
ols2, df2 = run_ols(X2, y, 'model2_state')

# Model 3: Model 2 + Year FE
X3 = pd.concat([X2, year_dummies], axis=1)
ols3, df3 = run_ols(X3, y, 'model3_year')

# Model 4: Model 3 + State-specific time trends
df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
state_trends = pd.DataFrame()
for col in state_dummies.columns:
    state_trends[col + '_trend'] = state_dummies[col] * df['year_numeric']
state_trends = state_trends.loc[valid_idx].astype(float)
X4 = pd.concat([X3, state_trends], axis=1)
ols4, df4 = run_ols(X4, y, 'model4_trend')

# ---- 10. IV regressions (price instrumented) ----
# Model 1 IV
iv1, df_iv1 = run_iv(y, X1, 'price', instruments, 'iv_model1_base_brand')
# Model 2 IV
iv2, df_iv2 = run_iv(y, X2, 'price', instruments, 'iv_model2_state')
# Model 3 IV
iv3, df_iv3 = run_iv(y, X3, 'price', instruments, 'iv_model3_year')
# Model 4 IV
iv4, df_iv4 = run_iv(y, X4, 'price', instruments, 'iv_model4_trend')

print("\nAll 8 regressions finished. Elasticities and ROI calculated for each model.")
