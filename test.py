import os
import networkx as nx
import walker
import matplotlib.pyplot as plt
import numpy as np
import time

graph_name = input("Graph file name: ")
graph_folder = graph_name + "-plots"

type_graph = int(input("Direct = 0, undirect = 1: "))

if(type_graph):
    G = nx.read_edgelist(f"graphs/{graph_name}.txt", create_using = nx.Graph)
else:
    G = nx.read_edgelist(f"graphs/{graph_name}.txt", create_using = nx.DiGraph)
print(G)

if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)
    print(f"{graph_folder} created.")

real_start_time = time.time()
pr = nx.pagerank(G, alpha = 0.85, max_iter=1000, tol=1e-05/G.number_of_nodes()) # Valore PageRank calcolato con il Power Method
real_end_time = time.time()
real_exe_time = real_end_time - real_start_time
print(f"Computation time real PageRank: {real_exe_time} s")

epsilon = 0.15

# Numero di random walks che partono da ogni nodo del grafo
K_values = [1, 2, 5, 10, 15, 20]

deviations = []  # Lista delle deviazioni per ogni valore di K
approx_execution_times = []

n_experiments = 10
max_deviations = {K: [] for K in K_values}
mean_deviations = {K: [] for K in K_values}
mean_execution_times = {K: [] for K in K_values}

approxPageRank_K = {}
pageRankIndices = np.array(list(pr.keys()))
pageRankValues = np.array(list(pr.values()))

fig, axs = plt.subplots(6, 1, figsize=(10, 24))
for i in range(n_experiments):
    j = 0
    for K in K_values:
        nx.set_node_attributes(G, 0, "nodeVisits")  # Inizializza il numero di visite di ogni nodo a 0
        # inizio delle random walks , X è un array NumPy delle walks
        # alpha = epsilon = probabilità di restart
        random_walks_start_time = time.time()
        X = walker.random_walks(G, n_walks = int(K), alpha = .15)
        random_walks_end_time = time.time()
        random_walks_exe_time = random_walks_end_time - random_walks_start_time
        print(f"Computation time random walks: {random_walks_exe_time} s")
        # Algoritmo di approssimazione Monte Carlo base per il PageRank
        approx_start_time = time.time()
        for walk in X:
            for node in walk:
                G.nodes[str(node)]["nodeVisits"] += 1
        approx_end_time = time.time()
        approx_exe_time = approx_end_time - approx_start_time
        print(f"Approx method execution time K = {K}: {approx_exe_time} s")
        mean_execution_times[K].append(approx_exe_time + random_walks_exe_time)

        approxPageRank = {}
        for v in G.nodes():
            approxPageRank[v] = (G.nodes[v]["nodeVisits"] * epsilon)/(K * G.number_of_nodes()) # kn = cnlogn

        diffPageRanks = [abs(pr[v] - approxPageRank[v]) for v in G.nodes()]
        mean_error = sum(diffPageRanks) / G.number_of_nodes()
        max_deviations[K].append(max(diffPageRanks))
        mean_deviations[K].append(mean_error)

        # Creo file e plot solo una volta
        if i == 0:
            deviations.append(mean_error)
            with open(f"{graph_folder}/approxPageRank" + str(K) + ".txt", "w") as file:
                for key, value in approxPageRank.items():
                    file.write(f"{key}: {value}\n")
            print("approxPageRank" + str(K) + ".txt updated.")

            # Indici ordinati per pageRank
            sortedNodes = np.argsort(pageRankValues, kind='heapsort')
            exactValues = pageRankValues[sortedNodes]
            approxValues = np.array([approxPageRank[pageRankIndices[i]] for i in sortedNodes])
            x = np.arange(len(sortedNodes))

            axs[j].plot(x, exactValues, label="PageRank Reale", color="blue", alpha=0.5)
            axs[j].plot(x, approxValues, label="PageRank Approssimato", color="red", alpha=0.5)

            axs[j].set_xlabel("Id Nodo")
            axs[j].set_ylabel("PageRank")
            axs[j].set_title(f"Valori esatti e approssimati di PageRank per K = {K} - Ordine crescente di PageRank esatto")
            axs[j].legend()
            axs[j].set_yscale('log')

            top_nodes_indices = sorted(range(len(approxValues)), key=lambda i: approxValues[i], reverse=True)[:100]
            top_nodes_pagerank = [approxValues[i] for i in top_nodes_indices]
            sorted_top_nodes = [sortedNodes[i] for i in top_nodes_indices]

            with open(f"{graph_folder}/top100PageRanks" + str(K) + ".txt", "w") as file:
                for node, pagerank in zip(sorted_top_nodes, top_nodes_pagerank):
                    file.write(f"Nodo: {node}\tPageRank: {pagerank}\n")
            print("top100PageRanks" + str(K) + ".txt updated.")

            approxPageRank_K[K] = approxPageRank

            j += 1

plt.tight_layout()
plt.savefig(f"{graph_folder}/pagerank_plots.png")
plt.show()

K_valid = []
for K in K_values:
    mean_approx_time = sum(mean_execution_times[K])
    print(f"Mean approx execution time K = {K}: {mean_approx_time / n_experiments}")
    if (mean_approx_time / n_experiments) < real_exe_time:
        print(f"Mean execution time for K = {K} is valid.")
        K_valid.append(K)

max_deviation_values = [np.mean(max_deviations[K]) for K in K_values]
max_deviation_std_values = [np.std(max_deviations[K]) for K in K_values]

mean_deviation_values = [np.mean(mean_deviations[K]) for K in K_values]
mean_deviation_std_values = [np.std(mean_deviations[K]) for K in K_values]

mean_execution_time_values = [np.mean(mean_execution_times[K]) for K in K_values]
execution_time_std_values = [np.std(mean_execution_times[K]) for K in K_values]

plt.figure()
plt.errorbar(K_values, max_deviation_values, yerr=max_deviation_std_values, marker='o', label='Deviazioni massime', capsize=10, color='red')
plt.xlabel("K")
plt.ylabel("Deviazione")
plt.title("Deviazioni Massime")
plt.savefig(f"{graph_folder}/max_deviations.png")
plt.show()
print("max_deviations.png created.")

plt.figure()
plt.errorbar(K_values, mean_deviation_std_values, yerr=mean_deviation_std_values, marker = 'o', label = 'Deviazioni medie', capsize=10)
plt.xlabel("K")
plt.ylabel("Deviazione")
plt.title("Deviazioni Medie")
plt.savefig(f"{graph_folder}/deviation_plot.png")
plt.show()
print("deviation_plot.png created.")

plt.figure()
plt.plot(K_values, [real_exe_time] * len(K_values), marker='o', linestyle='dashed', label='Tempo PageRank Reale', color='blue')
plt.errorbar(K_values, mean_execution_time_values, yerr=execution_time_std_values, marker='o', label='Tempi di esecuzione medi', capsize=10, color='red')
plt.xlabel("K")
plt.ylabel("Tempo di esecuzione (s)")
plt.title("Tempi di esecuzione medi")
plt.savefig(f"{graph_folder}/mean_execution_times.png")
plt.show()
print("mean_execution_times.png created.")

with open(f"{graph_folder}/realPageRank.txt", "w") as file:
    for key, value in pr.items():
        file.write(f"{key}: {value}\n")
print("realPageRank.txt updated.")

top_nodes_indices = sorted(range(len(exactValues)), key=lambda i: exactValues[i], reverse=True)[:100]
top_nodes_pagerank = [exactValues[i] for i in top_nodes_indices]
sorted_top_nodes = [sortedNodes[i] for i in top_nodes_indices]

with open(f"{graph_folder}/top100PageRanks.txt", "w") as file:
    for node, pagerank in zip(sorted_top_nodes, top_nodes_pagerank):
        file.write(f"Nodo: {node}\tPageRank: {pagerank}\n")
print("top100PageRanks.txt updated.")

mean_errors = {K: deviation for K, deviation in zip(K_values, deviations)}

with open(f"{graph_folder}/meanErrors.txt", "w") as file:
    for K, mean_error in mean_errors.items():
        file.write(f"K = {K}: {mean_error}\n")
print("meanErrors.txt updated.")

# Prepara gli array per il grafico
x = np.arange(1, len(sorted_top_nodes) + 1)
y_real = np.array(top_nodes_pagerank)

# Creazione grafico top 100
plt.figure()
fig, ax = plt.subplots(figsize = (20, 10))
plt.scatter(x, y_real, marker = 'o', s = 50, label='PageRank reale')
for K in K_values:
    if K in K_valid:
        y_approx = np.array([approxPageRank_K[K][pageRankIndices[node]] for node in sorted_top_nodes])
        plt.scatter(x, y_approx, marker='o', s=50, alpha=0.5, label=f'PageRank Approssimato (K = {K})')
ax.set_xticks(x)
ax.set_xticklabels(sorted_top_nodes, rotation = 90)
plt.xlabel("Ranking")
plt.ylabel("PageRank")
plt.title("PageRank dei Top 100 nodi")
plt.yscale('log')
plt.tight_layout()
ax.legend()
plt.savefig(f"{graph_folder}/real_pagerank_top100.png")
plt.show()
print("real_pagerank_top100.png created.")

# plt.figure()
# plt.plot(K_values, deviations, marker = 'o', label = 'Deviazioni medie')
# plt.xlabel("K")
# plt.ylabel("Deviazione")
# plt.title("Deviazioni al variare di K")
# plt.savefig(f"{graph_folder}/deviation_plot.png")
# plt.show()
# print("deviation_plot.png created.")
