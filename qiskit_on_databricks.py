# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# Quantum Computing with Qiskit: From Theory to Practice

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Quantum Computing: A Quick Recap
# MAGIC 
# MAGIC Quantum computing harnesses the principles of quantum mechanics to process information in fundamentally different ways than classical computers. Here's what makes it special:
# MAGIC 
# MAGIC ### Key Concepts:
# MAGIC - **Superposition**: Unlike classical bits (0 or 1), quantum bits (qubits) can exist in multiple states simultaneously
# MAGIC - **Entanglement**: Qubits can be correlated in ways that have no classical counterpart
# MAGIC - **Quantum Interference**: Quantum states can interfere with each other, leading to enhanced or diminished probabilities
# MAGIC 
# MAGIC ### Where Quantum Computing Makes Sense:
# MAGIC - Cryptography and security (e.g., Shor's algorithm for factoring large numbers)
# MAGIC - Optimization problems (e.g., traveling salesman problem)
# MAGIC - Quantum chemistry simulations
# MAGIC - Machine learning (quantum neural networks, quantum feature spaces)
# MAGIC 
# MAGIC ### Where Classical Computing Still Wins:
# MAGIC - Day-to-day computing tasks
# MAGIC - Tasks requiring precise, deterministic outcomes
# MAGIC - Problems that don't benefit from quantum parallelism
# MAGIC - Applications requiring stable, error-free computation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Introduction to Qiskit
# MAGIC 
# MAGIC Qiskit is an open-source framework for quantum computing developed by IBM. It provides tools for:
# MAGIC - Creating and manipulating quantum circuits
# MAGIC - Running simulations on classical computers
# MAGIC - Executing programs on real quantum computers
# MAGIC - Visualizing quantum states and results
# MAGIC 
# MAGIC ### Setting Up Qiskit Account
# MAGIC 1. Visit [IBM Quantum Experience](https://quantum-computing.ibm.com)
# MAGIC 2. Create a free account
# MAGIC 3. Get your API token from the user settings
# MAGIC 
# MAGIC Let's install and set up Qiskit:

# COMMAND ----------

# Install required packages
%pip install qiskit qiskit[visualization] pylatexenc

# COMMAND ----------

# Import necessary libraries
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Your First Quantum Circuit: Creating a Bell State
# MAGIC 
# MAGIC We'll start with creating a Bell state, one of the simplest examples of quantum entanglement.

# COMMAND ----------

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Add gates to create a Bell state
qc.h(0)  # Hadamard gate on qubit 0
qc.cx(0, 1)  # CNOT gate with control qubit 0 and target qubit 1

# Add measurement
qc.measure([0,1], [0,1])

# Draw the circuit
qc.draw(output='mpl')

# COMMAND ----------

# Execute the circuit on a simulator
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc, simulator, shots=1000)
result = job.result()

# Get the counts of measurement outcomes
counts = result.get_counts(qc)

# Plot the histogram
plot_histogram(counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Real-World Application: Quantum Random Number Generator
# MAGIC 
# MAGIC Let's create a true random number generator using quantum superposition.

# COMMAND ----------

def quantum_random_number(bits=4):
    # Create a quantum circuit with specified number of qubits
    qc = QuantumCircuit(bits, bits)
    
    # Put all qubits in superposition
    for i in range(bits):
        qc.h(i)
    
    # Measure all qubits
    qc.measure(range(bits), range(bits))
    
    # Execute the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1)
    result = job.result()
    
    # Convert binary output to decimal
    counts = result.get_counts(qc)
    binary = list(counts.keys())[0]
    return int(binary, 2)

# Generate some random numbers
random_numbers = [quantum_random_number() for _ in range(10)]
print(f"Generated random numbers: {random_numbers}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Exploring Quantum Interference
# MAGIC 
# MAGIC Let's create a simple interference experiment using the Mach-Zehnder interferometer analog.

# COMMAND ----------

# Create interferometer circuit
qc_interference = QuantumCircuit(1, 1)

# First beam splitter
qc_interference.h(0)

# Phase shifter
qc_interference.p(np.pi/4, 0)

# Second beam splitter
qc_interference.h(0)

# Measure
qc_interference.measure(0, 0)

# Execute with 1000 shots
job = execute(qc_interference, simulator, shots=1000)
result = job.result()
counts = result.get_counts(qc_interference)

# Plot results
plot_histogram(counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Key Takeaways and Future Directions
# MAGIC 
# MAGIC - Quantum computing offers unique capabilities for specific types of problems
# MAGIC - Tools like Qiskit make quantum experimentation accessible
# MAGIC - Current limitations include decoherence and error rates
# MAGIC - Active areas of research: error correction, quantum advantage demonstration
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - Explore more complex quantum algorithms
# MAGIC - Try running circuits on real quantum computers
# MAGIC - Investigate quantum machine learning applications
# MAGIC - Study quantum error correction techniques