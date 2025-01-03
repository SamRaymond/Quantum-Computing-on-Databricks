# Databricks notebook source
# MAGIC %md
# MAGIC # Quantum Computing with Qiskit: From Theory to Practice

# COMMAND ----------

# MAGIC %md
# MAGIC ##### !! ~~ NOTE: These APIs are still under development as of Jan 2025 and may have breaking changes ~~!!
# MAGIC
# MAGIC **Tested on MLR 16.1 single node**
# MAGIC

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

# Install dependencies
%pip install qiskit qiskit_aer qiskit_ibm_runtime numpy pylatexenc python-dotenv

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Creating Your First Quantum Circuit: The Bell State
# MAGIC
# MAGIC ## Introduction to Bell States
# MAGIC A Bell state represents one of the simplest yet most profound quantum phenomena - quantum entanglement. First proposed by Einstein, Podolsky, and Rosen in 1935, Bell states demonstrate the "spooky action at a distance" that exemplifies quantum mechanics.
# MAGIC
# MAGIC ## Building the Circuit: IBM's 4-Step Approach
# MAGIC
# MAGIC ### 1. Map: Design and Construction
# MAGIC We'll map our problem to quantum circuits using Qiskit's building blocks:
# MAGIC - Two qubits initialized in state |0⟩
# MAGIC - A Hadamard gate to create superposition
# MAGIC - A CNOT gate to create entanglement
# MAGIC - Measurement operations to observe results
# MAGIC
# MAGIC **Key Components:**
# MAGIC - Circuit Library: For basic quantum gates
# MAGIC - Quantum Info Library: For state visualization
# MAGIC - Custom Gates (if needed)
# MAGIC
# MAGIC ### 2. Optimize: Circuit Transpilation
# MAGIC Before running on real quantum hardware, we need to optimize our circuit:
# MAGIC - Use Qiskit's transpiler to convert to hardware-specific gates
# MAGIC - Apply optimization passes for efficiency
# MAGIC - Consider noise mitigation strategies
# MAGIC
# MAGIC **Optimization Tools:**
# MAGIC - Basic Transpiler
# MAGIC - AI-Enhanced Transpilation options
# MAGIC - Hardware-specific optimizations
# MAGIC
# MAGIC ### 3. Execute: Running the Circuit
# MAGIC We'll execute our circuit using multiple approaches:
# MAGIC - Simulation for testing
# MAGIC - Real quantum hardware for actual results
# MAGIC - Different execution modes for various purposes
# MAGIC
# MAGIC **Execution Options:**
# MAGIC - Local simulator
# MAGIC - IBM Quantum runtime primitives
# MAGIC - Various execution modes (shot-based, statevector, etc.)
# MAGIC
# MAGIC ### 4. Post-Process: Analyzing Results
# MAGIC Finally, we'll analyze and visualize our results:
# MAGIC - State tomography
# MAGIC - Visualization of quantum states
# MAGIC - Statistical analysis of measurements
# MAGIC
# MAGIC **Analysis Tools:**
# MAGIC - Quantum Info Library for state analysis
# MAGIC - Visualization modules for plotting
# MAGIC - Classical post-processing techniques
# MAGIC
# MAGIC ## Expected Outcomes
# MAGIC After running the circuit, we should observe:
# MAGIC - Perfect correlations between measurements
# MAGIC - Approximately 50/50 distribution between |00⟩ and |11⟩ states
# MAGIC - Evidence of quantum entanglement

# COMMAND ----------

# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator  
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import numpy as np

# COMMAND ----------

from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API token from the environment variable
IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")

service = QiskitRuntimeService(channel="ibm_quantum",token=IBM_QUANTUM_TOKEN)

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

# Get our backend and generate optimized circuit
backend = service.backend(name="ibm_brisbane")
# Convert to an ISA circuit and layout-mapped observables.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
 
isa_circuit.draw("mpl", idle_wires=False)

# COMMAND ----------

# Set up six different observables to verify entanglement
observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
observables = [SparsePauliOp(label) for label in observables_labels]

# COMMAND ----------

# MAGIC %md
# MAGIC ## To run on a simulator

# COMMAND ----------

 # Use the following code instead if you want to run on a simulator:
 
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
backend = FakeAlmadenV2()
estimator = Estimator(backend)
 
# Convert to an ISA circuit and layout-mapped observables.
 
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]
 
job = estimator.run([(isa_circuit, mapped_observables)])
result = job.result()
 
# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).
 
job_result = job.result()
 
# This is the result from our single pub, which had five observables,
# so contains information on all five.
 
pub_result = job.result()[0]

# COMMAND ----------

# Plot the result
 
from matplotlib import pyplot as plt
 
values = pub_result.data.evs
 
errors = pub_result.data.stds
 
# plotting graph
plt.plot(observables_labels, values, "-o")
plt.xlabel("Observables")
plt.ylabel("Values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## To run on the real Quantum Computer

# COMMAND ----------

# Get our backend and generate optimized circuit
backend = service.backend(name="ibm_brisbane")
# Convert to an ISA circuit and layout-mapped observables.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
 
isa_circuit.draw("mpl", idle_wires=False)
# Construct the Estimator instance.
estimator = Estimator(mode=backend)
estimator.options.resilience_level = 1
estimator.options.default_shots = 100
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]

# One pub, with one circuit to run against five different observables.
job = estimator.run([(isa_circuit, mapped_observables)])
 
# Use the job ID to retrieve your job data later
print(f">>> Job ID: {job.job_id()}")

# COMMAND ----------

# You can see the status of your run on the IBM Dashboard too:
# https://quantum.ibm.com/workloads

# COMMAND ----------

# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).
job_result = job.result()
 
# This is the result from our single pub, which had six observables,
# so contains information on all six.
pub_result = job.result()[0]

# COMMAND ----------

# Plot the result
 
from matplotlib import pyplot as plt
 
values = pub_result.data.evs
 
errors = pub_result.data.stds
 
# plotting graph
plt.plot(observables_labels, values, "-o")
plt.xlabel("Observables")
plt.ylabel("Values")
plt.show()
