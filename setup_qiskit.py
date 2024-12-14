import qiskit
from qiskit import IBMQ, QuantumCircuit, Aer, execute
from qiskit.providers.ibmq import AccountError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify installation:
print("Qiskit version:", qiskit.__version__)

# Load IBM Quantum account using the token from environment variables
try:
    IBMQ.load_account()
    print("IBM Quantum account loaded successfully.")
except AccountError:
    api_token = os.getenv('IBM_QUANTUM_TOKEN')
    if api_token:
        IBMQ.save_account(api_token)
        IBMQ.load_account()
        print("IBM Quantum account saved and loaded successfully.")
    else:
        print("IBM Quantum API token not found. Please set it in the .env file.")

# Get the provider and list backends
provider = IBMQ.get_provider()
backends = provider.backends()
print("Available backends:")
for backend in backends:
    print(f"- {backend.name()} - {backend.status().pending_jobs} pending jobs")

# Create a simple quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Quantum circuit created:")
print(qc)

# Execute the circuit on the Qiskit Aer simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
results = job.result()
counts = results.get_counts(qc)
print("Simulation results:")
print(counts)

