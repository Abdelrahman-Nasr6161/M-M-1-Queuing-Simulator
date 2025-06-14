import matplotlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class MM1Queue:
    def __init__(self, arrival_rate, service_rate, sim_time=10_000):
        self.lambda_rate = arrival_rate / 60
        self.mu_rate = service_rate / 60
        self.sim_time = sim_time
        self.time = 0
        self.queue = deque()
        self.server_busy = False
        self.server_end = 0
        self.served = 0
        self.total_system_time = 0
        self.total_queue_time = 0
        self.total_busy_time = 0
        self.state_times = {}
        self.last_change = 0
        self.customers_in_system = 0
        self.customer_log = []

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

            if self.server_busy and self.time >= self.server_end:
                self.customers_in_system -= 1
                self.served += 1

                if customers and customers[-1]['queue_time'] == 0:
                    customer = customers[-1]
                    system_time = self.time - customer['arrival']
                    self.total_system_time += system_time
                    self.total_queue_time += customer['queue_time']
                    service_start = self.time - system_time + customer['queue_time']
                    self.customer_log.append({
                        'arrival_time': customer['arrival'],
                        'queue_time': customer['queue_time'],
                        'service_start_time': service_start,
                        'departure_time': self.time
                    })

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
                    self.customer_log.append({
                        'arrival_time': arrival_time,
                        'queue_time': queue_time,
                        'service_start_time': self.time,
                        'departure_time': self.time + service_time
                    })
                else:
                    self.server_busy = False

            self.update_state()

        return self.get_metrics()

    def get_metrics(self):
        avg_in_system = sum(state * time for state, time in self.state_times.items()) / self.sim_time
        avg_in_queue = avg_in_system - (self.total_busy_time / self.sim_time)
        probs = {state: time / self.sim_time for state, time in self.state_times.items()}
        avg_time_in_system = self.total_system_time / self.served if self.served > 0 else 0
        avg_time_in_queue = self.total_queue_time / self.served if self.served > 0 else 0
        return {
            'customers_served': self.served,
            'total_system_time': self.total_system_time,
            'total_queue_time': self.total_queue_time,
            'server_busy_time': self.total_busy_time,
            'avg_customers_system': avg_in_system,
            'avg_customers_queue': avg_in_queue,
            'avg_time_in_system': avg_time_in_system,
            'avg_time_in_queue': avg_time_in_queue,
            'state_probabilities': probs,
            'customer_log': self.customer_log,
            'server_busy_time_percentage': self.total_busy_time / self.sim_time,
            'server_free_time_percentage': 1 - self.total_busy_time / self.sim_time
        }

def run_simulation(arrival_rate, service_rate):
    np.random.seed(42)
    sim = MM1Queue(arrival_rate, service_rate)
    return sim.run()

scenarios = [
    {"name": "Original (ρ = 0.33)", "lambda": 4, "mu": 12},
    {"name": "Scenario 1 (ρ = 0.5)", "lambda": 6, "mu": 12},
    {"name": "Scenario 2 (ρ = 0.9)", "lambda": 10.8, "mu": 12},
    {"name": "Scenario 2 (ρ = 0.1)", "lambda": 10, "mu": 100}
]
theoretical = []
practical = []

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
    print(f"Avg time in system per customer: {results['avg_time_in_system']:.3f} min")
    print(f"Avg time in queue per customer: {results['avg_time_in_queue']:.3f} min")
    print(f"Proportion spent busy: {results['server_busy_time_percentage']:.3f}")
    print(f"Proportion spent free: {results['server_free_time_percentage']:.3f}")
    print("State Proportions:")
    for state in sorted(results['state_probabilities'].keys())[:6]:
        prob = results['state_probabilities'][state]
        print(f"  P({state}): {prob:.3f}")

    print("\nFirst 10 customer logs:")
    for customer in results['customer_log'][:10]:
        print(f"Arrival: {customer['arrival_time']:.2f}, "
              f"Start: {customer['service_start_time']:.2f}, "
              f"Departure: {customer['departure_time']:.2f}, "
              f"Queue: {customer['queue_time']:.2f}")

    theoretical.append(scenario['lambda'] / scenario['mu'])
    practical.append(results['server_busy_time_percentage'])

plt.figure(figsize=(10, 6))
labels = ["Original", "Scenario 1", "Scenario 2", "Scenario 3"]
x = np.arange(len(labels))
plt.plot(x, [r for r in theoretical], marker='o', label='Theoretical (ρ)')
plt.plot(x, practical, marker='s', label='Practical (Server Busy %)')
plt.xticks(x, labels)
plt.ylabel('Proportion Server Busy')
plt.title('Theoretical vs Practical Server Busy Proportion')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
width = 0.35
plt.bar(x - width/2, theoretical, width, label='Theoretical (ρ)')
plt.bar(x + width/2, practical, width, label='Practical (Server Busy %)')
plt.xticks(x, labels)
plt.ylabel('Proportion Server Busy')
plt.title('Theoretical vs Practical Server Busy Proportion')
plt.legend()
plt.tight_layout()
plt.show()

print(theoretical)
print(practical)
