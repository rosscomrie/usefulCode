import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_quality_assessor import DataValidator, CheckLevel

# Create sample dataset with various data quality issues
def create_sample_data():
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
    
    # Introduce some data quality issues
    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1000)],  # Primary key
        'transaction_date': dates,
        'amount': np.random.normal(1000, 500, 1000),
        'email': [f'customer_{i}@email.com' for i in range(1000)],
        'status': np.random.choice(['active', 'inactive', 'pending'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    
    # 1. Duplicate PKs
    df.loc[10:15, 'customer_id'] = 'CUST_0001'
    
    # 2. Null values
    df.loc[20:25, 'transaction_date'] = None
    df.loc[30:35, 'amount'] = None
    df.loc[40:45, 'email'] = None
    
    # 3. Empty strings
    df.loc[50:55, 'email'] = ''
    
    # 4. Large numeric values
    df.loc[60:65, 'amount'] = 1e25
    
    # 5. Old dates
    df.loc[70:75, 'transaction_date'] = datetime(2010, 1, 1)
    
    return df

# Example 1: Basic validation with population-level checks
def example_population_checks():
    print("\nExample 1: Population-level checks")
    print("-" * 50)
    
    config = {
        'datetime': {
            'enabled': True,
            'level': CheckLevel.POPULATION,
            'max_days_old': 45
        },
        'numeric': {
            'enabled': True,
            'level': CheckLevel.POPULATION
        }
    }
    
    validator = DataValidator(config)
    df = create_sample_data()
    
    result = validator.verify_data(df)  # No PK specified
    
    print("Population-level validation results:")
    print(f"Valid: {result.is_valid}")
    print("\nMessages:")
    for msg in result.messages:
        print(f"- {msg}")
    print("\nKey Stats:")
    for key, value in result.stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

# Example 2: Primary key level validation
def example_pk_checks():
    print("\nExample 2: Primary key level checks")
    print("-" * 50)
    
    config = {
        'datetime': {
            'enabled': True,
            'level': CheckLevel.PRIMARY_KEY,
            'max_output': 3,  # Limit detailed messages
            'max_days_old': 45
        },
        'numeric': {
            'enabled': True,
            'level': CheckLevel.PRIMARY_KEY,
            'max_output': 3
        },
        'string': {
            'enabled': True,
            'level': CheckLevel.PRIMARY_KEY,
            'max_output': 3
        }
    }
    
    validator = DataValidator(config)
    df = create_sample_data()
    
    result = validator.verify_data(df, primary_key='customer_id')
    
    print("Primary key level validation results:")
    print(f"Valid: {result.is_valid}")
    print("\nMessages (limited to max_output per check):")
    for msg in result.messages:
        print(f"- {msg}")

# Example 3: Mixed validation levels
def example_mixed_checks():
    print("\nExample 3: Mixed validation levels")
    print("-" * 50)
    
    config = {
        'datetime': {
            'enabled': True,
            'level': CheckLevel.POPULATION,
            'max_days_old': 45
        },
        'numeric': {
            'enabled': True,
            'level': CheckLevel.PRIMARY_KEY,
            'max_output': 3
        },
        'string': {
            'enabled': True,
            'level': CheckLevel.ROW,
            'max_output': 3
        }
    }
    
    validator = DataValidator(config)
    df = create_sample_data()
    
    result = validator.verify_data(df, primary_key='customer_id')
    
    print("Mixed level validation results:")
    print(f"Valid: {result.is_valid}")
    print("\nMessages:")
    for msg in result.messages:
        print(f"- {msg}")

# Example 4: Validation with disabled checks
def example_disabled_checks():
    print("\nExample 4: Disabled checks")
    print("-" * 50)
    
    config = {
        'datetime': {
            'enabled': False  # Disable datetime checks
        },
        'numeric': {
            'enabled': True,
            'level': CheckLevel.POPULATION
        },
        'string': {
            'enabled': False  # Disable string checks
        }
    }
    
    validator = DataValidator(config)
    df = create_sample_data()
    
    result = validator.verify_data(df, primary_key='customer_id')
    
    print("Validation results with disabled checks:")
    print(f"Valid: {result.is_valid}")
    print("\nMessages:")
    for msg in result.messages:
        print(f"- {msg}")

if __name__ == "__main__":
    # Run all examples
    example_population_checks()
    example_pk_checks()
    example_mixed_checks()
    example_disabled_checks()