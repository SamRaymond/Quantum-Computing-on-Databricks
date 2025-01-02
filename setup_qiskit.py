import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify installation:
print("Qiskit version:", qiskit.__version__)

service = QiskitRuntimeService(channel="ibm_quantum",token=os.getenv('IBM_QUANTUM_TOKEN')) # channel="ibm_quantum" or "ibm_cloud"


# IBM Quantum Access Plans Overview:

# Open Plan (Free):
# - 10 minutes monthly runtime on 100+ qubit systems
# - Great for learning and exploration
# - Includes Qiskit Runtime as a Service

# Pay-As-You-Go:
# - $96 USD per minute, billed per second
# - Flexible access to quantum computers
# - Includes technical support

# Dedicated Service:
# - Subscription-based with custom pricing
# - Exclusive access to entire quantum system
# - Advanced features and possible on-site installation

# Note: Researchers can apply for IBM Quantum Credits for additional access

print("\nCurrent IBM Quantum Access Plans:")
print("1. Open Plan: Free - 10 min/month")
print("2. Pay-As-You-Go: $96/minute")
print("3. Dedicated Service: Custom pricing\n")


backend = service.backend(name="ibm_brisbane") # ibm_kyiv, ibm_brisbane, ibm_sherbrooke

# QubitProperties describes key physical characteristics of a qubit:

# - T1 (Relaxation Time): Time for qubit to decay from excited to ground state
# - T2 (Dephasing Time): Time before qubit loses quantum coherence 
# - Frequency: Operating frequency of the qubit (typically in GHz range)

# Longer T1 and T2 times indicate better qubit quality and lower error rates.
# The frequency is used for qubit control and manipulation.
# Print properties for all qubits
print(f"Number of qubits: {backend.num_qubits}")

# Collect stats for all qubits
t1_times = []
t2_times = []
frequencies = []

for i in range(backend.num_qubits):
    properties = backend.qubit_properties(qubit=i)
    t1_times.append(properties.t1)
    t2_times.append(properties.t2)
    frequencies.append(properties.frequency)

print("\nQubit Statistics:")
print(f"T1 (relaxation time):")
print(f"  Average: {sum(t1_times)/len(t1_times):.6f} seconds")
print(f"  Min: {min(t1_times):.6f} seconds")
print(f"  Max: {max(t1_times):.6f} seconds")

print(f"\nT2 (dephasing time):")
print(f"  Average: {sum(t2_times)/len(t2_times):.6f} seconds") 
print(f"  Min: {min(t2_times):.6f} seconds")
print(f"  Max: {max(t2_times):.6f} seconds")

print(f"\nFrequency:")
print(f"  Average: {sum(frequencies)/len(frequencies):.2f} Hz")
print(f"  Min: {min(frequencies):.2f} Hz")
print(f"  Max: {max(frequencies):.2f} Hz")


