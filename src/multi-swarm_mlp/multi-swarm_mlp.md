# Training ANN with QPSO using Multi-Swarm
## Requirements
- GNU/Linux (Kali Linux in this case)
- Python 3.9.2
- Pyswarms
- QDPSO
- Scikit-learn 
- Numpy
- Matplotlip
- Opencv-python (CV2)
- Joblib
- Pickle
- Seaborn
- Mahotas

To install the requirements execute the [req](./req.txt) file or execute the following command:
```
$ pip install -r req.txt
```
Installing the scikit-learn package with this command:
```
$ pip install -U scikit-learn==1.0
```
## Techniques for Training the Multi-Layer Perceptron
The techniques used are described below:
  
1. [**Parallelization with Parameter Exchange (T1):**](./parallel_e.py)
    - **Description:** Similar to basic parallelization, but after each training session, the best parameters were exchanged between the swarms of the hidden and output layers.
    - **Objective:** To determine if inter-layer information exchange during training enhances convergence and solution quality.

2. [**Basic Parallelization (T2):**](./parallel.py)
    - **Description:** Parallelization was applied while training the hidden and output layers, with both swarms operating simultaneously yet independently.
    - **Objective:** To assess the improvement in training time without parameter exchange between swarms.

3. [**Concurrency (T3):**](./concurrencia.py)
    - **Description:** Concurrency was utilized, allowing swarms to exchange the best parameters during each iteration of the QDPSO.
    - **Objective:** To increase adaptability and accelerate convergence through continuous information updating between swarms.

4. [**Concurrency with Post-Training Parameter Exchange (T4):**](./concurrencia_e.py)
    - **Description:** This technique combined concurrency with parameter exchange after each training session, similar to technique 1.
    - **Objective:** To combine the benefits of constant parameter updates and periodic reinforcement of best practices among swarms.

5. [**Without Parallelization or Concurrency (T5):**](./no_parallel.py)
    - **Description:** The swarms worked sequentially, first the hidden layer and then the output layer, without any parameter exchange.
    - **Objective:** To establish a baseline to compare the impact of parallelization and concurrency techniques on network performance.

6. [**Without Parallelization/Concurrency but with Parameter Exchange (T6):**](./no_parallel_e.py)
    - **Description:** This technique is similar to technique five but incorporates the exchange of the best parameters after each training session, even though the swarms operated sequentially.
    - **Objective:** To evaluate the impact of parameter exchange in a non-parallel, non-concurrent environment.

7. [**One Swarm Without Parallelization/Concurrency (T7):**](./one_main.py)
    - **Description:** A classical method training with one swarm.