# Quantum Computing on Databricks

This project demonstrates how to set up and run quantum computing experiments using Qiskit on Databricks. The project is configured using [Poetry](https://python-poetry.org/) for dependency management.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting Up Qiskit](#setting-up-qiskit)
  - [Install Qiskit](#install-qiskit)
  - [Obtain IBM Quantum API Token](#obtain-ibm-quantum-api-token)
  - [Set Up Environment Variables](#set-up-environment-variables)
- [Running the Setup Script](#running-the-setup-script)
  - [Contents of `setup_qiskit.py`](#contents-of-setup_qiskitpy)
- [Using the Quantum Circuit](#using-the-quantum-circuit)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.7 or higher**: Install Python from the [official website](https://www.python.org/downloads/).
- **Poetry**: Install Poetry for dependency management by following [these instructions](https://python-poetry.org/docs/#installation).
- **IBM Quantum Account**: Sign up for a free account at [IBM Quantum](https://quantum-computing.ibm.com/).

## Installation

1. **Clone the Repository**

   Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/your-username/Quantum_Computing_on_Databricks.git
   cd Quantum_Computing_on_Databricks   ```

2. **Install Dependencies**

   Use Poetry to install the project's dependencies:
   ```bash
   poetry install   ```

## Setting Up Qiskit

### Install Qiskit

Add Qiskit to your project dependencies:

```bash
poetry add qiskit
```

### Obtain IBM Quantum API Token

To access IBM's quantum backends, you'll need your personal API token.

1. **Create an IBM Quantum Account**

   - Visit the [IBM Quantum](https://quantum-computing.ibm.com/) website.
   - Click **Sign In** and create a new account or sign in with your existing IBMid.

2. **Retrieve Your API Token**

   - After logging in, click on your username in the upper right corner.
   - Select **Account Settings** from the dropdown menu.
   - Copy the **API Token** provided.

### Set Up Environment Variables

For security reasons, it's best to store your API token in an environment variable.

1. **Install `python-dotenv`**

   Add `python-dotenv` to manage environment variables:

   ```bash
   poetry add python-dotenv
   ```

2. **Create a `.env` File**

   In the root directory of your project, create a `.env` file:

   ```bash
   touch .env
   ```

3. **Add Your API Token**

   ```ini:.env
   IBM_QUANTUM_TOKEN=your-api-token-here
   ```

   > **Note:** Replace `your-api-token-here` with the API token you copied earlier. Do **not** share this token publicly.

## Running the Setup Script

Run the `setup_qiskit.py` script to verify your installation and set up your IBM Quantum account:

```bash
poetry run python setup_qiskit.py
```

### Contents of `setup_qiskit.py`

```python:setup_qiskit.py
import qiskit
from qiskit import IBMQ, Aer, execute
from qiskit.providers.ibmq import AccountError
from qiskit import QuantumCircuit
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify installation
print("Qiskit version:", qiskit.__version__)

# Load IBM Quantum account using the token from environment variables
api_token = os.getenv('IBM_QUANTUM_TOKEN')
if api_token:
    try:
        IBMQ.save_account(api_token)
    except:
        pass  # Account already saved
    try:
        IBMQ.load_account()
        print("IBM Quantum account loaded successfully.")
    except AccountError as e:
        print(f"Error loading account: {e}")
else:
    print("IBM Quantum API token not found. Please set it in the .env file.")

# List available backends
try:
    provider = IBMQ.get_provider()
    backends = provider.backends()
    print("Available backends:")
    for backend in backends:
        status = backend.status()
        print(f"- {backend.name()} - {status.pending_jobs} pending jobs")
except Exception as e:
    print(f"Error retrieving backends: {e}")

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
```

## Using the Quantum Circuit

With the setup complete, you're ready to run quantum circuits and experiments.

- **Experiment with Quantum Circuits**: Modify `setup_qiskit.py` or create new scripts to design and run your quantum circuits.
- **Submit Jobs to Quantum Devices**: Use your IBM Quantum account to submit jobs to real quantum hardware.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.