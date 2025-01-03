# Databricks notebook source
# MAGIC %md
# MAGIC NOTE: The following is based on the [qGAN tutorial](https://github.com/qiskit-community/qiskit-machine-learning/blob/stable/0.8/docs/tutorials/04_torch_qgan.ipynb) in the qiskit-community github reworked to run smoothly on Databricks. 

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # A QuantumGAN Implementation with PyTorch
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This tutorial introduces step-by-step how to build a PyTorch-based Quantum Generative Adversarial Network algorithm.
# MAGIC
# MAGIC The tutorial is structured as follows:
# MAGIC
# MAGIC 1. [Introduction](#1.-Introduction)
# MAGIC 2. [Data and Represtation](#2.-Data-and-Representation)
# MAGIC 3. [Definitions of the Neural Networks](#3.-Definitions-of-the-Neural-Networks)
# MAGIC 4. [Setting up the Training Loop](#4.-Setting-up-the-Training-Loop)
# MAGIC 5. [Model Training](#5.-Model-Training)
# MAGIC 6. [Results: Cumulative Density Functions](#6.-Results:-Cumulative-Density-Functions)
# MAGIC 7. [Conclusion](#7.-Conclusion)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## _Quantum_ Machine Learning? Why/What/How? Is it even worth it?
# MAGIC **TL;DR: What Are Qubits Doing in This Code?**
# MAGIC
# MAGIC In this QuantumGAN, the generator is implemented as a quantum circuit, using qubits to replace parts of the PyTorch neural network. Instead of generating data with linear layers and activation functions, the quantum circuit uses qubits and quantum gates to encode and transform random inputs into quantum states. Sampling from these states produces the generator’s outputs.
# MAGIC
# MAGIC This setup differs from a regular GAN by leveraging the unique properties of qubits to naturally represent probabilistic distributions, which can be harder for classical networks to model efficiently. The discriminator remains classical, combining PyTorch and quantum-generated data in a hybrid quantum-classical optimization loop.
# MAGIC
# MAGIC
# MAGIC ### Why QuantumGANs Could Outperform Classical GANs
# MAGIC
# MAGIC For now, the main advantage of QGANs lies in their ability to natively represent and sample from complex probability distributions using qubits. Classical GANs typically start with simple inputs, like Gaussian noise, which require extensive transformation layers to approximate complex patterns. In contrast, QGANs use quantum circuits to encode probability distributions into the amplitudes of qubits. For example, a quantum generator with just 10 qubits can represent  2^{10} = 1,024  possible outcomes simultaneously, allowing it to capture far more complex correlations with fewer parameters than a classical generator.
# MAGIC
# MAGIC This efficiency becomes more pronounced as dimensions increase, where classical GANs face quadratic or exponential growth in compute cost. QGANs, by leveraging the exponential state space of qubits, provide a more scalable solution for high-dimensional spaces. However, current quantum hardware is noisy and limited in qubit count, which constrains their practicality. Today, classical GANs remain superior for most tasks, but QGANs hold the potential to outperform them as quantum computing technology advances, particularly for problems involving complex or high-dimensional distributions that are computationally expensive for classical methods.
# MAGIC
# MAGIC ### Why QuantumGANs Don’t Outperform Classical GANs (Yet)
# MAGIC
# MAGIC While QGANs show promise, their performance today is limited by the constraints of current quantum hardware. Quantum computers are noisy, with high error rates during computations, and are limited in the number of qubits available. This makes it challenging to scale QGANs to handle real-world, large-scale data. Additionally, classical GANs benefit from decades of optimization in hardware and software, providing more reliable and efficient solutions for most practical applications.
# MAGIC
# MAGIC Moreover, the hybrid quantum-classical optimization loop in QGANs introduces additional overhead, as quantum circuit execution is slower and requires interfacing with classical systems for gradient computation. This often negates any theoretical advantages of quantum sampling for most current use cases. In summary, while QGANs have exciting potential, classical GANs remain the better choice for practical tasks until quantum hardware and algorithms mature significantly.

# COMMAND ----------

!pip install torch qiskit qiskit_machine_learning pylatexenc

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Introduction
# MAGIC
# MAGIC The qGAN [1] is a hybrid quantum-classical algorithm used for generative modeling tasks. The algorithm uses the interplay of a quantum generator $G_{\theta}$, i.e., an ansatz (parametrized quantum circuit), and a classical discriminator $D_{\phi}$, a neural network, to learn the underlying probability distribution given training data.
# MAGIC
# MAGIC The generator and discriminator are trained in alternating optimization steps, where the generator aims at generating probabilities that will be classified by the discriminator as training data values (i.e, probabilities from the real training distribution), and the discriminator tries to differentiate between original distribution and probabilities from the generator (in other words, telling apart the real and generated distributions). The final goal is for the quantum generator to learn a representation for the target probability distribution.
# MAGIC The trained quantum generator can, thus, be used to load a quantum state which is an approximate model of the target distribution.
# MAGIC
# MAGIC **References:**
# MAGIC
# MAGIC [1] Zoufal et al., [Quantum Generative Adversarial Networks for learning and loading random distributions](https://www.nature.com/articles/s41534-019-0223-2)
# MAGIC
# MAGIC ### 1.1. qGANs for Loading Random Distributions
# MAGIC
# MAGIC Given $k$-dimensional data samples, we employ a quantum Generative Adversarial Network (qGAN) to learn a random distribution and to load it directly into a quantum state:
# MAGIC
# MAGIC $$ \big| g_{\theta}\rangle = \sum_{j=0}^{2^n-1} \sqrt{p_{\theta}^{j}}\big| j \rangle $$
# MAGIC
# MAGIC where $p_{\theta}^{j}$ describe the occurrence probabilities of the basis states $\big| j\rangle$.
# MAGIC
# MAGIC The aim of the qGAN training is to generate a state $\big| g_{\theta}\rangle$ where $p_{\theta}^{j}$, for $j\in \left\{0, \ldots, {2^n-1} \right\}$, describe a probability distribution that is close to the distribution underlying the training data $X=\left\{x^0, \ldots, x^{k-1} \right\}$.
# MAGIC
# MAGIC For further details please refer to [Quantum Generative Adversarial Networks for Learning and Loading Random Distributions](https://arxiv.org/abs/1904.00043) _Zoufal, Lucchi, Woerner_ \[2019\].
# MAGIC
# MAGIC For an example of how to use a trained qGAN in an application, the pricing of financial derivatives, please see the
# MAGIC [Option Pricing with qGANs](https://qiskit-community.github.io/qiskit-finance/tutorials/10_qgan_option_pricing.html#) tutorial.
# MAGIC
# MAGIC ## 2. Data and Representation
# MAGIC
# MAGIC First, we need to load our training data $X$.
# MAGIC
# MAGIC In this tutorial, the training data is given by a 2D multivariate normal distribution.
# MAGIC
# MAGIC The goal of the generator is to learn how to represent such distribution, and the trained generator should correspond to an $n$-qubit quantum state
# MAGIC \begin{equation}
# MAGIC |g_{\text{trained}}\rangle=\sum\limits_{j=0}^{k-1}\sqrt{p_{j}}|x_{j}\rangle,
# MAGIC \end{equation}
# MAGIC where the basis states $|x_{j}\rangle$ represent the data items in the training data set
# MAGIC $X={x_0, \ldots, x_{k-1}}$ with $k\leq 2^n$ and $p_j$ refers to the sampling probability
# MAGIC of $|x_{j}\rangle$.
# MAGIC
# MAGIC To facilitate this representation, we need to map the samples from the multivariate
# MAGIC normal distribution to discrete values. The number of values that can be represented
# MAGIC depends on the number of qubits used for the mapping.
# MAGIC Hence, the data resolution is defined by the number of qubits.
# MAGIC If we use $3$ qubits to represent one feature, we have $2^3 = 8$ discrete values.
# MAGIC
# MAGIC We first begin by fixing seeds in the random number generators for reproducibility of the outcome in this tutorial.
# MAGIC

# COMMAND ----------

import torch
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 123456
_ = torch.manual_seed(123456)  # suppress output


# COMMAND ----------

# We fix the number of dimensions, the discretization number and compute the number of qubits required as $2^3 = 8$.

import numpy as np
num_dim = 2
num_discrete_values = 8
num_qubits = num_dim * int(np.log2(num_discrete_values))


# COMMAND ----------

# Now, we prepare a discrete distribution from the continuous 2D normal distribution.
# We evaluate the continuous probability density function (PDF) on the grid $(-2, 2)^2$ with a discretization of 8 values per feature. 
# Thus, we have $64$ values of the PDF. Since this will be a discrete distribution we normalize the obtained probabilities.
from scipy.stats import multivariate_normal

# Create a grid of coordinates from -2 to 2 with 8 discrete values
coords = np.linspace(-2, 2, num_discrete_values)

# Define a bivariate normal distribution with mean [0, 0] and identity covariance matrix
rv = multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 1]], seed=algorithm_globals.random_seed)

# Generate grid elements for evaluating the PDF
grid_elements = np.transpose([np.tile(coords, len(coords)), np.repeat(coords, len(coords))])

# Evaluate the PDF on the grid elements
prob_data = rv.pdf(grid_elements)

# Normalize the probabilities to sum to 1
prob_data = prob_data / np.sum(prob_data)

# COMMAND ----------

# Let's visualize our distribution. It is a nice bell-shaped bivariate normal distribution on a discrete grid.

import matplotlib.pyplot as plt
from matplotlib import cm

mesh_x, mesh_y = np.meshgrid(coords, coords)
grid_shape = (num_discrete_values, num_discrete_values)

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "3d"})
prob_grid = np.reshape(prob_data, grid_shape)
surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Definitions of the Neural Networks
# MAGIC In this section we define two neural networks as described above:
# MAGIC
# MAGIC - A quantum generator as a quantum neural network.
# MAGIC - A classical discriminator as a PyTorch-based neural network.
# MAGIC
# MAGIC ### 3.1. Definition of the quantum neural network ansatz
# MAGIC
# MAGIC Now, we define the parameterized quantum circuit $G\left(\boldsymbol{\theta}\right)$ with $\boldsymbol{\theta} = {\theta_1, ..., \theta_k}$ which will be used in our quantum generator.
# MAGIC
# MAGIC The quantum generator's architecture is analogous to a classical neural network, but implemented using quantum gates. Here's the detailed breakdown:
# MAGIC The generator uses what's called a "hardware efficient ansatz" - this is a quantum circuit design pattern that's optimized for actual quantum hardware limitations while still maintaining expressivity. It consists of:
# MAGIC
# MAGIC **Input Layer**: Takes a uniform distribution as input (analogous to random noise input in classical GANs)
# MAGIC **Circuit Architecture**:
# MAGIC
# MAGIC Uses alternating layers of:
# MAGIC - RY (rotation around Y-axis) gates: Adjusts amplitudes
# MAGIC - RZ (rotation around Z-axis) gates: Adjusts phases
# MAGIC - CX (controlled-NOT) gates: Creates entanglement between qubits
# MAGIC
# MAGIC Repeats this pattern 6 times (similar to having 6 hidden layers)
# MAGIC
# MAGIC _**Circuit Depth Considerations**_:
# MAGIC
# MAGIC - When the dimension k > 1, the circuit needs sufficient depth to capture complex probability distributions
# MAGIC - Like in classical deep learning, shallow circuits (analogous to shallow networks) can only learn simple patterns
# MAGIC - The deep structure (6 repetitions) provides enough parameters to learn and represent sophisticated probability distributions
# MAGIC - This is similar to how deeper neural networks can represent more complex functions
# MAGIC
# MAGIC The main difference from classical GANs is that instead of using matrix multiplications and nonlinear activations, we use quantum gates that operate on quantum states, allowing us to directly manipulate probability amplitudes in quantum superposition.

# COMMAND ----------

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

# Create a quantum circuit with the specified number of qubits
qc = QuantumCircuit(num_qubits)

# Apply Hadamard gates to all qubits to create superposition
qc.h(qc.qubits)

# Create an EfficientSU2 ansatz with the specified number of qubits and 6 repetitions
ansatz = EfficientSU2(num_qubits, reps=6)

# Compose the ansatz with the quantum circuit
qc.compose(ansatz, inplace=True)

# COMMAND ----------

# Let's draw our circuit and see what it looks like. On the plot we may notice a pattern that appears $6$ times.

qc.decompose().draw(output="mpl", style="clifford")

# COMMAND ----------

# Let's print the number of trainable parameters.
print(f"The number of trainable parameters in the quantum circuit is: {qc.num_parameters}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Definition of the quantum generator
# MAGIC
# MAGIC We start defining the generator by creating a sampler for the ansatz. The reference implementation is a statevector-based implementation, thus it returns exact probabilities as a result of circuit execution. In this case the implementation samples probabilities from the multinomial distribution constructed from the measured quasi probabilities. 
# MAGIC

# COMMAND ----------

from qiskit.primitives import StatevectorSampler as Sampler

sampler = Sampler()

# COMMAND ----------

# Next, we define a function that creates the quantum generator from a given parameterized quantum circuit. 
# Inside this function we create a neural network that returns the quasi probability distribution evaluated by the underlying Sampler.
#  We fix `initial_weights` for reproducibility purposes. 
# In the end we wrap the created quantum neural network in `TorchConnector` to make use of PyTorch-based training.


# COMMAND ----------

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

def create_generator() -> TorchConnector:
    # Create a SamplerQNN using the provided quantum circuit and sampler
    qnn = SamplerQNN(
        circuit=qc,  # Quantum circuit to be used
        sampler=sampler,  # Sampler instance
        input_params=[],  # Input parameters for the QNN
        weight_params=qc.parameters,  # Weight parameters for the QNN
        sparse=False,  # Use dense representation
    )

    # Generate initial weights for the QNN
    initial_weights = algorithm_globals.random.random(qc.num_parameters)
    
    # Wrap the QNN in a TorchConnector for PyTorch-based training
    return TorchConnector(qnn, initial_weights)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Definition of the classical discriminator
# MAGIC
# MAGIC Next, we define a PyTorch-based classical neural network that represents the classical discriminator. The underlying gradients can be automatically computed with PyTorch.
# MAGIC

# COMMAND ----------

from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        # Define the first linear layer with input size and 20 output features
        self.linear_input = nn.Linear(input_size, 20)
        # Define a LeakyReLU activation function with negative slope of 0.2
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Define the second linear layer with 20 input features and 1 output feature
        self.linear20 = nn.Linear(20, 1)
        # Define a Sigmoid activation function for the output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pass the input through the first linear layer
        x = self.linear_input(input)
        # Apply the LeakyReLU activation function
        x = self.leaky_relu(x)
        # Pass the result through the second linear layer
        x = self.linear20(x)
        # Apply the Sigmoid activation function to get the final output
        x = self.sigmoid(x)
        return x

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. Create a generator and a discriminator 
# MAGIC Now we create a generator and a discriminator.
# MAGIC

# COMMAND ----------

generator = create_generator()
discriminator = Discriminator(num_dim)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Setting up the Training Loop
# MAGIC In this section we set up:
# MAGIC
# MAGIC - A loss function for the generator and discriminator.
# MAGIC - Optimizers for both.
# MAGIC - A utility plotting function to visualize training process.
# MAGIC
# MAGIC ### 4.1. Definition of the loss functions
# MAGIC We want to train the generator and the discriminator with binary cross entropy as the loss function:
# MAGIC $$L\left(\boldsymbol{\theta}\right)=\sum_jp_j\left(\boldsymbol{\theta}\right)\left[y_j\log(x_j) + (1-y_j)\log(1-x_j)\right],$$
# MAGIC where $x_j$ refers to a data sample and $y_j$ to the corresponding label.
# MAGIC
# MAGIC Since PyTorch's `binary_cross_entropy` is not differentiable with respect to weights, we implement the loss function manually to be able to evaluate gradients.
# MAGIC

# COMMAND ----------

def adversarial_loss(input, target, w):
    bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
    weighted_loss = w * bce_loss
    total_loss = -torch.sum(weighted_loss)
    return total_loss


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2. Definition of the optimizers
# MAGIC In order to train the generator and discriminator, we need to define optimization schemes. In the following, we employ a momentum based optimizer called Adam, see [Kingma et al., Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980) for more details.
# MAGIC

# COMMAND ----------

from torch.optim import Adam

# Learning rate for the optimizers
lr = 0.01

# First and second momentum parameters for Adam optimizer
b1 = 0.7
b2 = 0.999

# Optimizer for the generator with weight decay for regularization
generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)

# Optimizer for the discriminator with weight decay for regularization
discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3. Visualization of the training process
# MAGIC We will visualize what is happening during the training by plotting the evolution of the generator's and the discriminator's loss functions during the training, as well as the progress in the relative entropy between the trained and the target distribution. We define a function that plots the loss functions and relative entropy. We call this function once an epoch of training is complete.
# MAGIC
# MAGIC Visualization of the training process begins when training data is collected across two epochs.
# MAGIC

# COMMAND ----------

from IPython.display import clear_output


def plot_training_progress():
    # we don't plot if we don't have enough data
    if len(generator_loss_values) < 2:
        return

    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Generator Loss
    ax1.set_title("Loss")
    ax1.plot(generator_loss_values, label="generator loss", color="royalblue")
    ax1.plot(discriminator_loss_values, label="discriminator loss", color="magenta")
    ax1.legend(loc="best")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.grid()

    # Relative Entropy
    ax2.set_title("Relative entropy")
    ax2.plot(entropy_values)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Relative entropy")
    ax2.grid()

    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Training
# MAGIC In the training loop we monitor not only loss functions, but relative entropy as well. The relative entropy describes a distance metric for distributions. Hence, we can use it to benchmark how close/far away the trained distribution is from the target distribution.
# MAGIC
# MAGIC Now, we are ready to train our model. It may take some time to train the model so be patient.
# MAGIC

# COMMAND ----------

import time
from scipy.stats import multivariate_normal, entropy

n_epochs = 50

num_qnn_outputs = num_discrete_values**num_dim

generator_loss_values = []
discriminator_loss_values = []
entropy_values = []

start = time.time()
for epoch in range(n_epochs):

    valid = torch.ones(num_qnn_outputs, 1, dtype=torch.float)
    fake = torch.zeros(num_qnn_outputs, 1, dtype=torch.float)

    # Configure input
    real_dist = torch.tensor(prob_data, dtype=torch.float).reshape(-1, 1)

    # Configure samples
    samples = torch.tensor(grid_elements, dtype=torch.float)
    disc_value = discriminator(samples)

    # Generate data
    gen_dist = generator(torch.tensor([])).reshape(-1, 1)

    # Train generator
    generator_optimizer.zero_grad()
    generator_loss = adversarial_loss(disc_value, valid, gen_dist)

    # store for plotting
    generator_loss_values.append(generator_loss.detach().item())

    generator_loss.backward(retain_graph=True)
    generator_optimizer.step()

    # Train Discriminator
    discriminator_optimizer.zero_grad()

    real_loss = adversarial_loss(disc_value, valid, real_dist)
    fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
    discriminator_loss = (real_loss + fake_loss) / 2

    # Store for plotting
    discriminator_loss_values.append(discriminator_loss.detach().item())

    discriminator_loss.backward()
    discriminator_optimizer.step()

    entropy_value = entropy(gen_dist.detach().squeeze().numpy(), prob_data)
    entropy_values.append(entropy_value)

    plot_training_progress()

elapsed = time.time() - start
print(f"Fit in {elapsed:0.2f} sec")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results: Cumulative Density Functions
# MAGIC In this section we compare the cumulative distribution function (CDF) of the trained distribution to the CDF of the target distribution.
# MAGIC
# MAGIC First, we generate a new probability distribution with PyTorch autograd turned off as we are not going to train the model anymore.
# MAGIC

# COMMAND ----------

with torch.no_grad():
    generated_probabilities = generator().numpy()


# COMMAND ----------

# And then, we plot the cumulative distribution functions of the generated distribution, original distribution, and the difference between them. Please, be careful, the scale on the third plot **is not the same** as on the first and second plot, and the actual difference between the two plotted CDFs is pretty small.


# COMMAND ----------

fig = plt.figure(figsize=(18, 9))

# Generated CDF
gen_prob_grid = np.reshape(np.cumsum(generated_probabilities), grid_shape)

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.set_title("Generated CDF")
ax1.plot_surface(mesh_x, mesh_y, gen_prob_grid, linewidth=0, antialiased=False, cmap=cm.coolwarm)
ax1.set_zlim(-0.05, 1.05)

# Real CDF
real_prob_grid = np.reshape(np.cumsum(prob_data), grid_shape)

ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.set_title("True CDF")
ax2.plot_surface(mesh_x, mesh_y, real_prob_grid, linewidth=0, antialiased=False, cmap=cm.coolwarm)
ax2.set_zlim(-0.05, 1.05)

# Difference
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.set_title("Difference between CDFs")
ax3.plot_surface(
    mesh_x, mesh_y, real_prob_grid - gen_prob_grid, linewidth=2, antialiased=False, cmap=cm.coolwarm
)
ax3.set_zlim(-0.05, 0.1)

plt.show()

# Explanation
displayHTML("""
<p>The first plot shows the Cumulative Distribution Function (CDF) of the generated probabilities. The second plot shows the CDF of the true probabilities. The third plot shows the difference between the two CDFs. Note that the scale on the third plot is different, indicating that the actual difference between the generated and true CDFs is small.</p>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusion
# MAGIC
# MAGIC Quantum generative adversarial networks employ the interplay of a generator and discriminator to map an approximate representation of a probability distribution underlying given data samples into a quantum channel. This tutorial presents a self-standing PyTorch-based qGAN implementation where the generator is given by a quantum channel, i.e., a variational quantum circuit, and the discriminator by a classical neural network, and discusses the application of efficient learning and loading of generic probability distributions into quantum states. The loading requires $\mathscr{O}\left(poly\left(n\right)\right)$ gates and can thus enable the use of potentially advantageous quantum algorithms.
