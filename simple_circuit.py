import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from dotenv import load_dotenv
import asyncio
import os 

# Load environment variables from .env file
load_dotenv()

# Verify installation:
print("Qiskit version:", qiskit.__version__)

service = QiskitRuntimeService(channel="ibm_quantum",token=os.getenv('IBM_QUANTUM_TOKEN')) # channel="ibm_quantum" or "ibm_cloud"


# IBM Quantum Backends:
# ibm_brisbane: 20 qubit system, 1000+ qubits
# ibm_kyiv: 20 qubit system, 1000+ qubits
# ibm_sherbrooke: 5 qubit system, 1000+ qubits

backend = service.backend(name="ibm_brisbane") # ibm_kyiv, ibm_brisbane, ibm_sherbrooke


# Two-qubit Bell state

# In this simple example, we will create a Bell state, which is a superposition of two qubits that are entangled.
 
# A brief introduction to quantum gates:
# Hadamard gate (H): Creates a superposition of 0 and 1 (|0> -> |0> + |1>)
# CNOT gate (CX): Entangles two qubits (|00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>)

#Create the circuit
qc = QuantumCircuit(2) # Create a quantum circuit with 2 qubits
qc.h(0) # Hadamard gate on qubit 0
qc.cx(0, 1) # CNOT gate with qubit 0 as control and qubit 1 as target
print(qc.draw())
#      ┌───┐     
# q_0: ┤ H ├──■──
#      └───┘┌─┴─┐
# q_1: ─────┤ X ├
#           └───┘

from qiskit.quantum_info import Pauli

# What are Pauli gates?
# Pauli gates are a set of three single-qubit gates that are used to manipulate the state of a qubit. They are named after Wolfgang Pauli, who first described them in 1926. The three Pauli gates are:
# X: Pauli-X gate, also known as the NOT gate. It flips the state of a qubit (|0> -> |1>, |1> -> |0>)
# Y: Pauli-Y gate. It rotates the state of a qubit around the Y-axis of the Bloch sphere (|0> -> i|1>, |1> -> -i|0>)
# Z: Pauli-Z gate. It rotates the state of a qubit around the Z-axis of the Bloch sphere (|0> -> |0>, |1> -> -|1>)

# Pauli gates are used to create and manipulate quantum states, and they are essential for quantum computing and quantum information theory.    

ZZ = Pauli('ZZ')
ZI = Pauli('ZI')
IZ = Pauli('IZ')
XX = Pauli('XX')
XI = Pauli('XI')
IX = Pauli('IX')
operators = [ZZ, ZI, IZ, XX, XI, IX]

# Convert Pauli operators to string labels
operator_labels = [str(op) for op in operators]

# Step 2: Optimize our observables
# [Assuming there are steps here if needed]

# Step 3: Execute the circuit with the Aer simulator
estimator = Estimator()
job = estimator.run([qc]*len(operators), operators)
result = job.result()
expectation_values = result.values

print(f"Expectation values for the Bell state for the Pauli operators:")
for label, value in zip(operator_labels, expectation_values):
    print(f"{label}: {value}")

# Step 4: Visualize the results
import matplotlib.pyplot as plt

# plt.bar(operator_labels, expectation_values)
# plt.xlabel('Pauli Operators')
# plt.ylabel('Expectation Values')
# plt.title('Expectation Values for the Bell State')
# plt.show()



# Now we can run this on 100 qubits, or the N-Qubit GHZ (Greenberger-Horne-Zeilinger) state 
# We can also run this on the IBM Quantum backend 
N = 100
def run_ghz_state(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    return qc

qc = run_ghz_state(N)
# print(ghz_state.draw())

# Let's look at how the distance between the qubits affects the expectation values, we'll use the ZZ operator

from qiskit.quantum_info import SparsePauliOp # We need sparse since we are using a large number of qubits

operator_strings = ["Z" + "I"*i  + "Z" + "I" * (N-2-i) for i in range(N-1)]

# print(operator_strings)
operators = [SparsePauliOp(op) for op in operator_strings]


# print(operators)

# Optimize th problem for quantum execution
# Transpile the circuit for the backend. This is a step that is used to optimize the circuit for the backend.
# Trasnspilation: The process of converting a quantum circuit into a form that is compatible with a specific quantum hardware.

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

backend_name = "ibm_brisbane"
backend = QiskitRuntimeService(channel="ibm_quantum",token=os.getenv('IBM_QUANTUM_TOKEN')).backend(name=backend_name)
pass_manager = generate_preset_pass_manager(optimization_level=1,backend=backend)


qc_transpiled = pass_manager.run(qc)
operators_transpiled = [op.apply_layout(qc_transpiled.layout) for op in operators]



from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import EstimatorOptions

options = EstimatorOptions()
options.resilience_level = 1 # This is the resilience level, which is a measure of the robustness of the circuit to errors.


estimator = Estimator(backend)

async def run_quantum_job():
    # Submit the quantum job to the IBM backend
    job = estimator.run([(qc_transpiled, operators_transpiled)])
    print(f"Submitted job with ID: {job.job_id()}")

    # Wait for the quantum job to complete
    print("Waiting for job to complete...")
    job_result = job.result()
    print("Job completed!")
    print(job_result)
    # Extract and process the results from the quantum computer
    expectation_values = job_result
    distances = range(N-1)  # distances between qubits

    # Print the results showing how quantum correlations change with distance
    print(f"\nExpectation values for the GHZ state for the ZZ operator at different distances:")
    for distance, value in zip(distances, expectation_values):
        print(f"Distance {distance}: {value:.3f}")

# Run the quantum job without waiting for asynchronous tasks
# asyncio.run(run_quantum_job())

# Alternatively, we can return the job_result and process it later
def return_job_result(run_ID):
    job = service.job(run_ID)
    job_result = job.result()
    return job_result


run_ID = "cxex1vz3ej4g008fytjg"
job_result = return_job_result(run_ID)
print(job_result)

