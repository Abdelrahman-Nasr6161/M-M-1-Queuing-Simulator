import numpy as np
from collections import deque

class MM1Queue:
    def __init__(self, arrival_rate, service_rate, sim_time=10000):
        self.lambda_rate = arrival_rate / 60
        self.mu_rate = service_rate / 60
        self.sim_time = sim_time
        
        self.time = 0
        self.queue = deque()
        self.server_busy = False
        self.server_end = 0
        
        # Metrics
        self.served = 0
        self.total_system_time = 0
        self.total_queue_time = 0
        self.total_busy_time = 0
        self.state_times = {}
        self.last_change = 0
        self.customers_in_system = 0
        
    def exp_time(self, rate):
        return np.random.exponential(1 / rate)
    
    def update_state(self):
        duration = self.time - self.last_change
        if self.customers_in_system not in self.state_times:
            self.state_times[self.customers_in_system] = 0
        self.state_times[self.customers_in_system] += duration
        self.last_change = self.time
    
    def run(self):
        next_arrival = self.exp_time(self.lambda_rate)
        customers = []
        
        while self.time < self.sim_time:
            if self.server_busy:
                next_event = min(next_arrival, self.server_end)
            else:
                next_event = next_arrival
            
            self.update_state()
            self.time = next_event
            
            # Arrival
            if self.time == next_arrival and self.time < self.sim_time:
                arrival_time = self.time
                self.customers_in_system += 1
                
                if not self.server_busy:
                    service_time = self.exp_time(self.mu_rate)
                    self.server_busy = True
                    self.server_end = self.time + service_time
                    customers.append({'arrival': arrival_time, 'queue_time': 0})
                else:
                    self.queue.append(arrival_time)
                
                next_arrival = self.time + self.exp_time(self.lambda_rate)
            
            # Departure
            if self.server_busy and self.time >= self.server_end:
                self.customers_in_system -= 1
                self.served += 1
                
                if customers and customers[-1]['queue_time'] == 0:
                    customer = customers[-1]
                    system_time = self.time - customer['arrival']
                    self.total_system_time += system_time
                    self.total_queue_time += customer['queue_time']
                
                busy_duration = self.server_end - (self.server_end - self.exp_time(self.mu_rate))
                self.total_busy_time += busy_duration
                
                if self.queue:
                    arrival_time = self.queue.popleft()
                    queue_time = self.time - arrival_time
                    service_time = self.exp_time(self.mu_rate)
                    self.server_end = self.time + service_time
                    
                    customers.append({'arrival': arrival_time, 'queue_time': queue_time})
                    system_time = queue_time + service_time
                    self.total_system_time += system_time
                    self.total_queue_time += queue_time
                else:
                    self.server_busy = False
        
        self.update_state()
        return self.get_metrics()
    
    def get_metrics(self):
        # Time-averaged customers in system
        avg_in_system = sum(state * time for state, time in self.state_times.items()) / self.sim_time
        
        # Time-averaged customers in queue
        avg_in_queue = avg_in_system - (self.total_busy_time / self.sim_time)
        
        # State probabilities
        probs = {state: time/self.sim_time for state, time in self.state_times.items()}
        
        return {
            'customers_served': self.served,
            'total_system_time': self.total_system_time,
            'total_queue_time': self.total_queue_time,
            'server_busy_time': self.total_busy_time,
            'avg_customers_system': avg_in_system,
            'avg_customers_queue': avg_in_queue,
            'state_probabilities': probs
        }

def run_simulation(arrival_rate, service_rate):
    np.random.seed(42)
    sim = MM1Queue(arrival_rate, service_rate)
    return sim.run()

# Run simulations for different scenarios
scenarios = [
    {"name": "Original (ρ = 0.33)", "lambda": 4, "mu": 12},
    {"name": "Scenario 1 (ρ = 0.5)", "lambda": 6, "mu": 12},
    {"name": "Scenario 2 (ρ = 0.9)", "lambda": 10.8, "mu": 12}
]

for scenario in scenarios:
    print(f"\n{scenario['name']}: λ = {scenario['lambda']}, μ = {scenario['mu']}")
    print("-" * 40)
    
    results = run_simulation(scenario['lambda'], scenario['mu'])
    
    print(f"Customers served: {results['customers_served']}")
    print(f"Total system time: {results['total_system_time']:.1f} min")
    print(f"Total queue time: {results['total_queue_time']:.1f} min") 
    print(f"Server busy time: {results['server_busy_time']:.1f} min")
    print(f"Avg customers in system: {results['avg_customers_system']:.3f}")
    print(f"Avg customers in queue: {results['avg_customers_queue']:.3f}")
    print("State probabilities:")
    for state in sorted(results['state_probabilities'].keys())[:6]:
        prob = results['state_probabilities'][state]
        print(f"  P({state}): {prob:.3f}")