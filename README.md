🛍️ Customer Segmentation using K-Means Clustering

This repository contains my solution for Task 02 of my Machine Learning Internship at **SkillCraft Technology**. The project focuses on grouping retail store customers based on their purchase history to help the marketing team plan targeted strategies.



## 📋 Project Overview
The goal is to perform **Customer Segmentation** using an unsupervised machine learning algorithm. By identifying patterns in customer data, we can categorize them into different groups based on their spending behavior and income levels.

### Key Objectives:
* Use **Unsupervised Learning** to find hidden patterns in data.
* Determine the optimal number of clusters using the **Elbow Method**.
* Visualize the segments to identify "Target Customers" for marketing campaigns.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `pandas` & `numpy` for data processing.
    * `scikit-learn` for the K-Means algorithm.
    * `matplotlib` & `seaborn` for cluster visualization.

---

## 📊 Dataset Description
The dataset consists of 200 mall customers with the following features:
* **CustomerID:** Unique ID for each customer.
* **Age:** Age of the customer.
* **Annual Income (k$):** Annual income in thousands of dollars.
* **Spending Score (1-100):** Score assigned by the mall based on customer behavior.

---

## 🚀 Implementation Steps

### 1. The Elbow Method
To find the optimal value of $k$ (number of clusters), I calculated the Within-Cluster Sum of Squares (WCSS). The "elbow" point was found at **$k=5$**.



### 2. Cluster Analysis
The resulting clusters are:
1. **Target Group:** High income, High spending (Focus for marketing).
2. **Sensible:** High income, Low spending.
3. **Standard:** Average income, Average spending.
4. **Careless:** Low income, High spending.
5. **Frugal:** Low income, Low spending.

---

## 📁 Repository Structure
* `Mall_Customers.csv`: The dataset.
* `task2_clustering.py`: Python implementation script.
* `README.md`: Project documentation.

## 📬 Contact
Feel free to reach out if you have any questions about this project!

MANISH KUMAR
