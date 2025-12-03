import pandas as pd
import numpy as np

def generate_data(n_samples=1000):
    np.random.seed(42)
    
    cpu_request = np.random.randint(100, 4000, n_samples)
    mem_request = np.random.randint(100, 8000, n_samples)
    cpu_limit = cpu_request + np.random.randint(100, 2000, n_samples)
    mem_limit = mem_request + np.random.randint(100, 4000, n_samples)
    runtime_minutes = np.random.randint(10, 1440, n_samples)
    
    controller_kinds = ['Deployment', 'StatefulSet', 'DaemonSet', 'Job']
    controller_kind = np.random.choice(controller_kinds, n_samples)
    
    # Simulate cpu_usage based on features with some noise
    # Base usage is a fraction of request + random noise
    base_usage = cpu_request * 0.6 + mem_request * 0.05 + runtime_minutes * 0.1
    noise = np.random.normal(0, 200, n_samples)
    
    # Adjust for controller kind
    kind_factor = {'Deployment': 1.0, 'StatefulSet': 1.2, 'DaemonSet': 0.8, 'Job': 1.1}
    kind_multiplier = np.array([kind_factor[k] for k in controller_kind])
    
    cpu_usage = (base_usage * kind_multiplier + noise).clip(min=0)
    
    # Ensure usage doesn't exceed limit significantly (soft limit behavior)
    cpu_usage = np.minimum(cpu_usage, cpu_limit * 1.1)

    df = pd.DataFrame({
        'cpu_request': cpu_request,
        'mem_request': mem_request,
        'cpu_limit': cpu_limit,
        'mem_limit': mem_limit,
        'runtime_minutes': runtime_minutes,
        'controller_kind': controller_kind,
        'cpu_usage': cpu_usage
    })
    
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv('data/cpu_usage.csv', index=False)
    print("Generated data/cpu_usage.csv")
