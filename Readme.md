# 🌲 Parallelizing Random Forest Training

## 📘 Course Information
**Course:** CSS 311 – Parallel and Distributed Computing  
**Department:** Computer Science, IIIT Kottayam  
**Instructor:** Dr. John Paul Martin  

---

## 👨‍💻 Authors
- **Gutta Yuva Sai Vinay** – Roll No: 2023BCD0044  
- **Koduri Siddhardha** – Roll No: 2023BCS0071  

---

## 🎯 Project Overview
This project aims to **speed up Random Forest model training** by implementing **parallelization using OpenMP**.  
Random Forest, being an ensemble of decision trees, is computationally expensive when trained serially.  
By leveraging **multi-core processors**, the training process is parallelized to achieve faster execution and improved efficiency.

---

## 🧩 Objectives
- Implement a **serial Random Forest** trainer.  
- Identify **parallelizable sections** of the algorithm.  
- Implement **OpenMP-based parallelization**.  
- Measure and compare:
  - Execution time  
  - Speedup  
  - Efficiency  
- Maintain identical model accuracy between serial and parallel versions.  

---

## ⚙️ Methodology
1. Load and preprocess dataset.  
2. Initialize Random Forest parameters.  
3. Train each decision tree:  
   - **Serial:** Sequential training  
   - **Parallel:** Concurrent training using OpenMP threads  
4. Combine all trained trees.  
5. Compare performance metrics between both versions.

---

## 🧮 Algorithm Summary

### 🧠 Serial Algorithm
**Steps:**
1. For each tree:
   - Draw bootstrap sample.
   - Select random features.
   - Train decision tree sequentially.
2. Combine all trees to form final model.  

**Time Complexity:**  
`O(N_trees × T_tree)`

---

### ⚡ Parallel Algorithm (Using OpenMP)
**Steps:**
1. Parallelize the tree training loop using OpenMP:  
   ```cpp
   #pragma omp parallel for
   for (int i = 0; i < N_trees; i++) {
       forest[i].train(dataset);
   }

Each thread trains one tree independently.

Combine trained trees after all threads finish.

Time Complexity:
O((N_trees × T_tree) / P)
where P = number of parallel threads (≤ CPU cores)

💻 Code Structure
📁 Parallel-Random-Forest/
├── RandomForest.cpp        # Serial and Parallel implementation
├── Pdcreport.docx          # Detailed report
├── Presentation.pptx       # 20-slide classroom presentation
├── README.md               # Project documentation
└── sample_dataset.csv      # (Optional) Example dataset

🧱 Requirements

Ensure your system supports OpenMP.

🖥️ Software:

C++ Compiler with OpenMP support

On Linux: g++ (recommended)

On Windows: MinGW with OpenMP enabled

⚙️ Compilation & Execution
▶️ Compile (Serial Version)
g++ RandomForest.cpp -o SerialRandomForest
./SerialRandomForest

⚡ Compile (Parallel Version with OpenMP)
g++ -fopenmp RandomForest.cpp -o ParallelRandomForest
./ParallelRandomForest

🧾 Example Output
Starting Parallel Random Forest Training...
Trained tree 1/10 on thread 0
Trained tree 2/10 on thread 1
Trained tree 3/10 on thread 2
Parallel Training Completed!

📊 Results Summary
Version	Execution Time	Speedup	Efficiency
Serial	High	1×	100%
Parallel	Low	1–2×	80–95%

Parallel version reduces training time significantly.

Maintains same accuracy as serial version.

Speedup depends on number of trees, CPU cores, and workload size.

🚧 Challenges Faced

Synchronization between threads.

Shared memory handling in OpenMP.

Balancing workload among threads.

Diminishing returns after full core usage.

Debugging and validating parallel execution.

🔍 Observations

Small datasets → minimal improvement.

Large datasets → noticeable speedup.

Minor thread overhead during setup.

Achieved good scalability with moderate workloads.

🧠 Conclusion

Successfully parallelized Random Forest training using OpenMP.

Achieved faster execution and improved scalability.

Accuracy and model behavior remain consistent with serial version.

Demonstrates benefits of task-level parallelism in ML workloads.

🚀 Future Scope

Extend to MPI for distributed system training.

Explore GPU-based parallelization (CUDA/OpenCL).

Improve load balancing and memory optimization.

Apply to larger, real-world machine learning datasets.

🧾 References

Quinn, M. J. Parallel Programming in C with MPI and OpenMP, McGraw-Hill, 2004.

OpenMP Architecture Review Board. OpenMP Application Program Interface v5.0, 2023.

Tanenbaum, A. S. Distributed Systems: Principles and Paradigms, Pearson, 2019.