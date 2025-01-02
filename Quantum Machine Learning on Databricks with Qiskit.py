# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # Quantum Machine Learning with Qiskit: Hamiltonian Approaches and Hybrid Systems
# MAGIC 
# MAGIC ## Introduction to Quantum-Classical Hybrid Systems
# MAGIC 
# MAGIC Hybrid quantum-classical systems combine the advantages of both quantum and classical computing:
# MAGIC - **Quantum Part**: Handles specific computations that quantum computers excel at (like simulating quantum systems or exploring large solution spaces)
# MAGIC - **Classical Part**: Manages optimization, pre/post-processing, and coordination of the quantum operations
# MAGIC 
# MAGIC The Hamiltonian approach is fundamental in quantum computing as it describes the total energy of a quantum system and its evolution over time.

# COMMAND ----------
# First, let's set up our environment and verify our installation
# Install required packages if not already present
%pip install qiskit qiskit-aer qiskit-ibm-runtime sklearn python-dotenv matplotlib

# COMMAND ----------
# Import all necessary libraries
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import qiskit
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Verify installation
print("Qiskit version:", qiskit.__version__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setting up IBM Quantum Access
# MAGIC 
# MAGIC Before proceeding, make sure you have:
# MAGIC 1. Created an IBM Quantum account at quantum-computing.ibm.com
# MAGIC 2. Generated your API token
# MAGIC 3. Added your token to your Databricks secrets or environment variables
# MAGIC 
# MAGIC Available backends:
# MAGIC - ibm_brisbane: 20 qubit system
# MAGIC - ibm_kyiv: 20 qubit system
# MAGIC - ibm_sherbrooke: 5 qubit system

# COMMAND ----------
# Load environment variables and setup backend
load_dotenv()

# Initialize the quantum service
service = QiskitRuntimeService(
    channel="ibm_quantum",
    token=os.getenv('IBM_QUANTUM_TOKEN')
)

# Select the backend - we'll use Brisbane for this example
backend = service.backend(name="ibm_brisbane")

# Print available backends
print("Available backends:")
for b in service.backends():
    print(f"- {b.name}: {b.configuration().n_qubits} qubits")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Understanding Parametrized Quantum Circuits
# MAGIC 
# MAGIC ### Why We Need Parametrization
# MAGIC 
# MAGIC Traditional quantum circuits with fixed gates are great for quantum algorithms like Shor's or Grover's, but they have limitations when it comes to machine learning tasks:
# MAGIC 
# MAGIC 1. **Fixed vs. Learnable Operations**:
# MAGIC    - Fixed circuits only give us measurements based on quantum physics principles
# MAGIC    - We need a way to "learn" from data, not just measure quantum states
# MAGIC 
# MAGIC 2. **Bridging Classical and Quantum**:
# MAGIC    - Regular quantum circuits don't provide a mechanism to encode classical information
# MAGIC    - Parametrized gates act as an interface between classical data and quantum operations

# COMMAND ----------
# Let's create a simple parametrized circuit
n_qubits = 2  # We'll start with 2 qubits
circuit = QuantumCircuit(n_qubits)

# Create parameters
theta1 = Parameter('θ1')
theta2 = Parameter('θ2')

# Build parametrized circuit
circuit.rx(theta1, 0)  # Parametrized X rotation on first qubit
circuit.rx(theta2, 1)  # Parametrized X rotation on second qubit
circuit.cnot(0, 1)    # Entangle the qubits

# Draw the circuit
print("Our parametrized circuit:")
circuit.draw()

# COMMAND ----------
# Let's try different parameter values
param_values1 = {theta1: np.pi/4, theta2: np.pi/2}
param_values2 = {theta1: np.pi, theta2: np.pi/4}

# Create bound circuits
bound_circuit1 = circuit.bind_parameters(param_values1)
bound_circuit2 = circuit.bind_parameters(param_values2)

print("Circuit with first set of parameters:")
bound_circuit1.draw()
print("\nCircuit with second set of parameters:")
bound_circuit2.draw()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Creating a Simple Ising Hamiltonian
# MAGIC 
# MAGIC The Ising model is one of the most fundamental models in statistical mechanics and quantum computing.
# MAGIC It describes a system of spins that can point either up or down and interact with their neighbors.

# COMMAND ----------
# Create a simple Ising Hamiltonian
n_qubits = 3  # Let's use 3 qubits
h_strength = 1.0  # Strength of the transverse field

# Create ZZ interactions between neighboring qubits
zz_terms = []
for i in range(n_qubits-1):
    pauli_str = ['I'] * n_qubits
    pauli_str[i] = 'Z'
    pauli_str[i+1] = 'Z'
    zz_terms.append(''.join(pauli_str))

# Create X field terms
x_terms = []
for i in range(n_qubits):
    pauli_str = ['I'] * n_qubits
    pauli_str[i] = 'X'
    x_terms.append(''.join(pauli_str))

# Combine terms with coefficients
all_terms = zz_terms + x_terms
coeffs = [1.0] * len(zz_terms) + [h_strength] * len(x_terms)

# Create the Hamiltonian
H = SparsePauliOp(all_terms, coeffs)

print("Ising Hamiltonian terms:")
for term, coeff in zip(all_terms, coeffs):
    print(f"{coeff:.1f} * {term}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Quantum Neural Network Implementation
# MAGIC 
# MAGIC Now let's create a simple quantum neural network and test it with some data.
# MAGIC We'll use it for binary classification.

# COMMAND ----------
# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Dataset shape:", X.shape)
print("Number of training samples:", len(X_train))
print("Number of test samples:", len(X_test))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Advanced QML Implementation
# MAGIC 
# MAGIC Now we'll implement a more sophisticated Quantum Neural Network with:
# MAGIC - Proper gradient-based training
# MAGIC - Async execution for efficient hardware usage
# MAGIC - Expectation value computation
# MAGIC - Hybrid quantum-classical optimization

# COMMAND ----------
# First, let's create our quantum neural network structure
class QuantumNeuralNetwork:
    def __init__(self, n_qubits, backend):
        self.n_qubits = n_qubits
        self.backend = backend
        self.estimator = Estimator(mode=backend)
        
    def create_quantum_circuit(self, x, params):
        """Creates a quantum circuit with data encoding and trainable parameters."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # Data encoding layer
        for i in range(self.n_qubits):
            qc.rx(x[i], i)
        
        # Trainable rotation layer
        for i in range(self.n_qubits):
            qc.ry(params[i], i)
        
        # Entangling layer
        for i in range(self.n_qubits-1):
            qc.cnot(i, i+1)
            
        qc.measure_all()
        return qc

# COMMAND ----------
# Let's create a training step that we can visualize
async def train_step(qnn, x, y, params, learning_rate=0.1):
    """Single training step with visualization"""
    grad = np.zeros_like(params)
    epsilon = 0.01
    
    # Calculate gradients
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        circuit_plus = qnn.create_quantum_circuit(x, params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= epsilon
        circuit_minus = qnn.create_quantum_circuit(x, params_minus)
        
        # Draw the circuits for visualization
        print(f"Circuit for parameter {i} plus epsilon:")
        print(circuit_plus.draw())
        print(f"\nCircuit for parameter {i} minus epsilon:")
        print(circuit_minus.draw())
        
        # Calculate expectation values
        result_plus = await qnn.estimator.run([circuit_plus], [H]).result()
        result_minus = await qnn.estimator.run([circuit_minus], [H]).result()
        
        grad[i] = (result_plus.values[0] - result_minus.values[0]) / (2 * epsilon)
    
    # Update and return new parameters
    return params - learning_rate * grad

# COMMAND ----------
# Initialize our QNN and run a training experiment
n_qubits = 4  # Match our number of features
qnn = QuantumNeuralNetwork(n_qubits=n_qubits, backend=backend)

# Initialize random parameters
initial_params = np.random.random(n_qubits) * 2 * np.pi
print("Initial parameters:", initial_params)

# Let's visualize the initial circuit
initial_circuit = qnn.create_quantum_circuit(X_train[0], initial_params)
print("\nInitial circuit structure:")
print(initial_circuit.draw())

# COMMAND ----------
# Run a mini training loop for demonstration
async def run_mini_training(qnn, n_steps=3):
    params = initial_params
    loss_history = []
    
    for step in range(n_steps):
        print(f"\nTraining step {step + 1}")
        
        # Use just one sample for demonstration
        x_sample = X_train[0]
        y_sample = y_train[0]
        
        # Perform training step
        params = await train_step(qnn, x_sample, y_sample, params)
        
        # Calculate and store loss
        circuit = qnn.create_quantum_circuit(x_sample, params)
        result = await qnn.estimator.run([circuit], [H]).result()
        loss = abs(result.values[0] - y_sample)
        loss_history.append(loss)
        
        print(f"Current loss: {loss:.4f}")
        print("Current parameters:", params)
        
    return params, loss_history

# COMMAND ----------
# Run our training demonstration
import asyncio

async def main():
    try:
        print("Starting mini training loop...")
        final_params, loss_history = await run_mini_training(qnn)
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('Training Loss over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()
        
        print("\nFinal parameters:", final_params)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        if 'session' in locals():
            session.close()

# Run the async function
asyncio.run(main())

# COMMAND ----------
# Create a simple classification circuit for comparison
    
    # Encode features
    for i in range(4):
        qc.rx(x_sample[i], i)
    
    # Add entangling layers
    for i in range(3):
        qc.cnot(i, i+1)
    
    # Add measurement
    qc.measure_all()
    
    return qc

# Let's try it with one sample
sample_circuit = create_classification_circuit(X_train[0])
print("Circuit for first training sample:")
sample_circuit.draw()

# COMMAND ----------
# Run the circuit on a simulator
simulator = Aer.get_backend('qasm_simulator')
job = simulator.run(sample_circuit, shots=1000)
result = job.result()
counts = result.get_counts()

print("Measurement outcomes distribution:")
for outcome, count in counts.items():
    print(f"{outcome}: {count} shots")

# Create a simple bar plot of the results
plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values())
plt.title('Measurement Outcomes Distribution')
plt.xlabel('Basis State')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Now that we've explored the basics of quantum circuits and Hamiltonians, you can:
# MAGIC 1. Experiment with different circuit architectures
# MAGIC 2. Try running on real quantum hardware
# MAGIC 3. Implement more sophisticated training loops
# MAGIC 4. Explore different data encoding strategies
# MAGIC 
# MAGIC Remember to always close your quantum sessions when you're done to free up resources!