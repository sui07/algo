
// 1.Write a program to implement radix Sort.

#include <stdio.h>
#include <stdlib.h>
void radix_sort(int arr[], int n) {
int i, exp = 1;
int max = arr[0];
int *output = malloc(sizeof(int) * n);
// Find the maximum element in the array
for (i = 1; i < n; i++) {
if (arr[i] > max) {
max = arr[i];
}
}
// Sort the elements based on each digit
while (max / exp > 0) {
int bucket[10] = {0};
// Count the number of elements in each bucket
for (i = 0; i < n; i++) {
bucket[arr[i] / exp % 10]++;
}
// Calculate the starting index for each bucket
for (i = 1; i < 10; i++) {
bucket[i] += bucket[i - 1];
}
// Place each element into its corresponding bucket
for (i = n - 1; i >= 0; i--) {
output[--bucket[arr[i] / exp % 10]] = arr[i];
}
// Copy the sorted elements back to the original array
for (i = 0; i < n; i++) {
arr[i] = output[i];
}
// Move to the next digit
exp *= 10;
}
free(output);
}
int main() {
int arr[] = {170, 45, 75, 90, 802, 24, 2, 66};
int n = sizeof(arr) / sizeof(arr[0]);

printf("Original array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
radix_sort(arr, n);
printf("Sorted array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
return 0;
}

// 2.Write a program to implement randomized quick Sort.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void swap(int *a, int *b) {
int temp = *a;
*a = *b;
*b = temp;
}
int partition(int arr[], int start, int end) {
int pivot = arr[end];
int i = start - 1;
for (int j = start; j <= end - 1; j++) {
if (arr[j] <= pivot) {
i++;
swap(&arr[i], &arr[j]);
}
}
swap(&arr[i + 1], &arr[end]);
return (i + 1);
}
int random_partition(int arr[], int start, int end) {
srand(time(NULL));
int random = start + rand() % (end - start + 1);
swap(&arr[random], &arr[end]);
return partition(arr, start, end);
}
void quicksort(int arr[], int start, int end) {
if (start < end) {
int pivot_index = random_partition(arr, start, end);
quicksort(arr, start, pivot_index - 1);
quicksort(arr, pivot_index + 1, end);
}
}
int main() {
int arr[] = {10, 7, 8, 9, 1, 5};
int n = sizeof(arr) / sizeof(arr[0]);
printf("Original array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
quicksort(arr, 0, n - 1);
printf("Sorted array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
return 0;
}

// 3.Write a program to implement marge Sort.

#include <stdio.h>
// Merge two sorted subarrays into a single sorted array
void merge(int arr[], int l, int m, int r) {
int i, j, k;
int n1 = m - l + 1;
int n2 = r - m;
// Create temp arrays
int L[n1], R[n2];
// Copy data to temp arrays
for (i = 0; i < n1; i++)
L[i] = arr[l + i];
for (j = 0; j < n2; j++)
R[j] = arr[m + 1 + j];
// Merge the temp arrays back into arr[l..r]
i = 0;
j = 0;
k = l;
while (i < n1 && j < n2) {
if (L[i] <= R[j]) {
arr[k] = L[i];
i++;
}
else {
arr[k] = R[j];
j++;
}
k++;
}
// Copy the remaining elements of L[], if there are any
while (i < n1) {
arr[k] = L[i];
i++;
k++;
}
// Copy the remaining elements of R[], if there are any
while (j < n2) {
arr[k] = R[j];
j++;
k++;
}
}
// Merge sort function
void mergeSort(int arr[], int l, int r) {
if (l < r) {
// Find the middle point
int m = l + (r - l) / 2;
// Sort first and second halves
mergeSort(arr, l, m);
mergeSort(arr, m + 1, r);

// Merge the sorted halves
merge(arr, l, m, r);
}
}
int main() {
int arr[] = { 12, 11, 13, 5, 6, 7 };
int arr_size = sizeof(arr) / sizeof(arr[0]);
printf("Given array is \n");
for (int i = 0; i < arr_size; i++)
printf("%d ", arr[i]);
mergeSort(arr, 0, arr_size - 1);
printf("\nSorted array is \n");
for (int i = 0; i < arr_size; i++)
printf("%d ", arr[i]);
return 0;
}

// 4.Write a program to implement insertion Sort.

#include <stdio.h>
void insertion_sort(int arr[], int n) {
int i, key, j;
for (i = 1; i < n; i++) {
key = arr[i];
j = i - 1;
while (j >= 0 && arr[j] > key) {
arr[j + 1] = arr[j];
j = j - 1;
}
arr[j + 1] = key;
}
}
int main() {
int arr[] = { 12, 11, 13, 5, 6 };
int n = sizeof(arr) / sizeof(arr[0]);
insertion_sort(arr, n);
printf("Sorted array: ");
for (int i = 0; i < n; i++)
printf("%d ", arr[i]);
printf("\n");
return 0;
}

// 5. Write a program to find shortest paths from a given vertex to other vertices using
Dijkstra's algorithm

#include <stdio.h>
#include <limits.h>
#define V 6

// Number of vertices in the graph

// Function to find the vertex with minimum distance value
int minDistance(int dist[], int sptSet[]) {
int min = INT_MAX, min_index;
for (int v = 0; v < V; v++)
if (sptSet[v] == 0 && dist[v] <= min)
min = dist[v], min_index = v;
return min_index;
}
// Function to print the shortest path from source to j
void printPath(int parent[], int j) {
// Base Case : If j is source
if (parent[j] == -1)
return;
printPath(parent, parent[j]);
printf("%d ", j);
}
// Function to print the shortest distance from source to all vertices
void printSolution(int dist[], int n, int parent[]) {
int src = 0;
printf("Vertex\t Distance\t Path");
for (int i = 1; i < V; i++) {
printf("\n%d -> %d \t\t %d\t\t%d ", src, i, dist[i], src);
printPath(parent, i);
}
}
// Function that implements Dijkstra's single source shortest path algorithm
void dijkstra(int graph[V][V], int src) {
int dist[V];
int sptSet[V];
int parent[V];
for (int i = 0; i < V; i++) {
parent[0] = -1;
dist[i] = INT_MAX;
sptSet[i] = 0;
}
dist[src] = 0;
for (int count = 0; count < V - 1; count++) {
int u = minDistance(dist, sptSet);
sptSet[u] = 1;
for (int v = 0; v < V; v++) {
if (!sptSet[v] && graph[u][v] && dist[u] + graph[u][v] < dist[v]) {

parent[v] = u;
dist[v] = dist[u] + graph[u][v];
}
}
}
printSolution(dist, V, parent);
}
int main() {
// Sample graph
int graph[V][V] = {
{0, 1, 4, 0, 0, 0},
{1, 0, 2, 5, 0, 0},
{4, 2, 0, 1, 6, 0},
{0, 5, 1, 0, 3, 7},
{0, 0, 6, 3, 0, 2},
{0, 0, 0, 7, 2, 0}
};
dijkstra(graph, 0);
return 0;
}

// 6.Write a program to determine the LCS of two given sequences.

#include <stdio.h>
#include <string.h>
#define MAX_LENGTH 100
int lcs(char* str1, char* str2, int len1, int len2) {
int i, j;
int dp[MAX_LENGTH][MAX_LENGTH]; // dp[i][j] stores the length of LCS of str1[0...i-1] and
str2[0...j-1]
// Initialize the first row and column of the dp array to 0
for (i = 0; i <= len1; i++) {
dp[i][0] = 0;
}
for (j = 0; j <= len2; j++) {
dp[0][j] = 0;
}
// Fill the remaining entries of the dp array
for (i = 1; i <= len1; i++) {
for (j = 1; j <= len2; j++) {
if (str1[i-1] == str2[j-1]) {
dp[i][j] = 1 + dp[i-1][j-1];
}
else {
dp[i][j] = (dp[i-1][j] > dp[i][j-1]) ? dp[i-1][j] : dp[i][j-1];
}
}
}
return dp[len1][len2];
}
int main() {
char str1[] = "ABCDGH";
char str2[] = "AEDFHR";
int len1 = strlen(str1);
int len2 = strlen(str2);
printf("Length of LCS is %d\n", lcs(str1, str2, len1, len2));
return 0;
}

// 7.Write a program to implement Depth-First Search in a graph.

#include <stdio.h>
#include <stdlib.h>
#define MAX_NODES 100
int graph[MAX_NODES][MAX_NODES];
int visited[MAX_NODES];
void dfs(int node, int num_nodes) {
visited[node] = 1;
printf("%d ", node);
for (int i = 0; i < num_nodes; i++) {
if (graph[node][i] && !visited[i]) {
dfs(i, num_nodes);
}
}
}
int main() {
int num_nodes, num_edges;
printf("Enter number of nodes and edges: ");
scanf("%d%d", &num_nodes, &num_edges);
printf("Enter edges (source destination):\n");
for (int i = 0; i < num_edges; i++) {
int u, v;
scanf("%d%d", &u, &v);
graph[u][v] = 1;
graph[v][u] = 1;
}
printf("DFS traversal: ");
for (int i = 0; i < num_nodes; i++) {
visited[i] = 0;
}
dfs(0, num_nodes);
return 0;
}

// 8.Write a program to implement Heap Sort.

#include <stdio.h>
#include <stdlib.h>
void swap(int *a, int *b) {
int temp = *a;
*a = *b;
*b = temp;
}
void heapify(int arr[], int n, int i) {
int largest = i;
int left_child = 2 * i + 1;
int right_child = 2 * i + 2;
if (left_child < n && arr[left_child] > arr[largest]) {
largest = left_child;
}
if (right_child < n && arr[right_child] > arr[largest]) {
largest = right_child;
}
if (largest != i) {
swap(&arr[i], &arr[largest]);
heapify(arr, n, largest);
}
}
void heapsort(int arr[], int n) {
for (int i = n / 2 - 1; i >= 0; i--) {
heapify(arr, n, i);
}
for (int i = n - 1; i > 0; i--) {
swap(&arr[0], &arr[i]);
heapify(arr, i, 0);
}
}
int main() {
int arr[] = {10, 7, 8, 9, 1, 5};
int n = sizeof(arr) / sizeof(arr[0]);
printf("Original array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
heapsort(arr, n);
printf("Sorted array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
return 0;
}

// 9. Write a program to print all the nodes reachable from a given starting node in a
digraph using BFS method.

#include <stdio.h>
#include <stdlib.h>
#define MAX_NODES 100
// Define a struct to represent a node in the graph
typedef struct Node {
int value;
struct Node* next;
} Node;
// Define a struct to represent the graph
typedef struct Graph {
int num_nodes;
Node* adj_list[MAX_NODES];
} Graph;
// Function to add an edge to the graph
void add_edge(Graph* graph, int src, int dest) {
// Create a new node for the destination vertex
Node* new_node = (Node*) malloc(sizeof(Node));
new_node->value = dest;
new_node->next = NULL;
// Add the new node to the adjacency list of the source vertex
if (graph->adj_list[src] == NULL) {
graph->adj_list[src] = new_node;
} else {
Node* curr_node = graph->adj_list[src];
while (curr_node->next != NULL) {
curr_node = curr_node->next;
}
curr_node->next = new_node;
}
}
// Function to print all nodes reachable from a given starting node using BFS
void bfs(Graph* graph, int start_node) {
int visited[MAX_NODES] = {0}; // Mark all nodes as unvisited
int queue[MAX_NODES], front = 0, rear = 0;
visited[start_node] = 1; // Mark the starting node as visited
queue[rear++] = start_node; // Add the starting node to the queue
printf("Nodes reachable from node %d: ", start_node);
while (front < rear) {
int curr_node = queue[front++]; // Dequeue the next node from the queue
printf("%d ", curr_node);
// Add all unvisited neighbors of the current node to the queue
Node* curr_adj_list = graph->adj_list[curr_node];
while (curr_adj_list != NULL) {
int neighbor = curr_adj_list->value;
if (!visited[neighbor]) {
visited[neighbor] = 1;

queue[rear++] = neighbor;
}
curr_adj_list = curr_adj_list->next;
}
}
printf("\n");
}
int main() {
Graph* graph = (Graph*) malloc(sizeof(Graph));
graph->num_nodes = 6;
for (int i = 0; i < graph->num_nodes; i++) {
graph->adj_list[i] = NULL;
}
// Add edges to the graph
add_edge(graph, 0, 1);
add_edge(graph, 0, 2);
add_edge(graph, 1, 2);
add_edge(graph, 2, 0);
add_edge(graph, 2, 3);
add_edge(graph, 3, 3);
add_edge(graph, 4, 5);
// Perform BFS starting from each node in the graph
for (int i = 0; i < graph->num_nodes; i++) {
bfs(graph, i);
}
return 0;
}

// 10.Write a program to perform Linear Search.

#include <stdio.h>
int linear_search(int arr[], int n, int x) {
for (int i = 0; i < n; i++) {
if (arr[i] == x) {
return i;
}
}
return -1;
}
int main() {
int arr[] = {10, 7, 8, 9, 1, 5};
int n = sizeof(arr) / sizeof(arr[0]);
int x = 8;
printf("Original array: ");
for (int i = 0; i < n; i++) {
printf("%d ", arr[i]);
}
printf("\n");
int index = linear_search(arr, n, x);
if (index == -1) {
printf("%d not found in the array.\n", x);
} else {
printf("%d found at index %d in the array.\n", x, index);
}
return 0;
}

// 11. Write a program to find Minimum Cost Spanning Tree of a given undirected graph using
Kruskal's algorithm.

// 12.Write a program to perform DFS traversal and mark visited vertices.

#include <stdio.h>
#include <stdlib.h>
#define MAX_VERTICES 100
// Adjacency list node
struct node {
int vertex;
struct node* next;
};
// Graph structure
struct Graph {
int num_vertices;
struct node** adj_lists;
int* visited;
};
// Create a new node
struct node* createNode(int v) {
struct node* newNode = (struct node*)malloc(sizeof(struct node));
newNode->vertex = v;
newNode->next = NULL;
return newNode;
}
// Add an edge to the graph
void addEdge(struct Graph* graph, int src, int dest) {
// Add edge from source to destination
struct node* newNode = createNode(dest);
newNode->next = graph->adj_lists[src];
graph->adj_lists[src] = newNode;
// Add edge from destination to source
newNode = createNode(src);
newNode->next = graph->adj_lists[dest];
graph->adj_lists[dest] = newNode;
}
// Depth-first search
void DFS(struct Graph* graph, int vertex) {
graph->visited[vertex] = 1;
printf("%d ", vertex);
// Traverse adjacency list of vertex
struct node* adj_list = graph->adj_lists[vertex];
while (adj_list != NULL) {
int adj_vertex = adj_list->vertex;
if (graph->visited[adj_vertex] == 0) {
DFS(graph, adj_vertex);
}
adj_list = adj_list->next;
}
}
int main() {
struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));

graph->num_vertices = 6;
// Allocate memory for adjacency lists and visited array
graph->adj_lists = (struct node**)malloc(graph->num_vertices * sizeof(struct node*));
graph->visited = (int*)malloc(graph->num_vertices * sizeof(int));
// Initialize adjacency lists and visited array
for (int i = 0; i < graph->num_vertices; i++) {
graph->adj_lists[i] = NULL;
graph->visited[i] = 0;
}
// Add edges to the graph
addEdge(graph, 0, 1);
addEdge(graph, 0, 2);
addEdge(graph, 1, 3);
addEdge(graph, 2, 4);
addEdge(graph, 3, 5);
// Perform DFS traversal
printf("DFS traversal: ");
DFS(graph, 0);
printf("\n");
// Free memory
for (int i = 0; i < graph->num_vertices; i++) {
struct node* adj_list = graph->adj_lists[i];
while (adj_list != NULL) {
struct node* temp = adj_list;
adj_list = adj_list->next;
free(temp);
}
}
free(graph->adj_lists);
free(graph->visited);
free(graph);
return 0;
}

// 13.Write a program to check whether a given graph is connected or not using DFS method.

#include <stdio.h>
#define MAX 10
int visited[MAX];
int graph[MAX][MAX];
int vertices;
void DFS(int vertex) {
visited[vertex] = 1;
for(int i = 1; i <= vertices; i++) {
if(graph[vertex][i] && !visited[i]) {
DFS(i);
}
}
}
int isConnected() {
for(int i = 1; i <= vertices; i++) {
visited[i] = 0;
}
DFS(1);
for(int i = 1; i <= vertices; i++) {
if(!visited[i]) {
return 0;
}
}
return 1;
}
int main() {
printf("Enter the number of vertices: ");
scanf("%d", &vertices);
printf("Enter the adjacency matrix:\n");
for(int i = 1; i <= vertices; i++) {
for(int j = 1; j <= vertices; j++) {
scanf("%d", &graph[i][j]);
}
}
if(isConnected()) {
printf("The graph is connected.\n");
}
else {
printf("The graph is not connected.\n");
}
return 0;
}

// 14.Write a program to implement All-Pair Shortest paths problem using Floyd's algorithm.

#include <stdio.h>
#include <limits.h>
#define V 4 // Number of vertices in the graph
void printSolution(int dist[][V]) {
printf("Shortest distances between every pair of vertices:\n");
for (int i = 0; i < V; i++) {
for (int j = 0; j < V; j++) {
if (dist[i][j] == INT_MAX)
printf("INF\t");
else
printf("%d\t", dist[i][j]);
}
printf("\n");
}
}
void floydWarshall(int graph[][V]) {
int dist[V][V];
for (int i = 0; i < V; i++) {
for (int j = 0; j < V; j++) {
dist[i][j] = graph[i][j];
}
}
for (int k = 0; k < V; k++) {
for (int i = 0; i < V; i++) {
for (int j = 0; j < V; j++) {
if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX &&
dist[i][k] + dist[k][j] < dist[i][j])
dist[i][j] = dist[i][k] + dist[k][j];
}
}
}
printSolution(dist);
}
int main() {
int graph[V][V] = {
{0, 5, INT_MAX, 10},
{INT_MAX, 0, 3, INT_MAX},
{INT_MAX, INT_MAX, 0, 1},
{INT_MAX, INT_MAX, INT_MAX, 0}
};
floydWarshall(graph);
return 0;
}

// 15. Write a program to find the maximum and minimum in a given list of n elements using
divide and conquer.

#include <stdio.h>
void findMinMax(int arr[], int low, int high, int *min, int *max) {
int mid, leftMin, leftMax, rightMin, rightMax;
// If there is only one element in the array
if (low == high) {
*min = arr[low];
*max = arr[low];
return;
}
// If there are two elements in the array
if (high == low + 1) {
if (arr[low] < arr[high]) {
*min = arr[low];
*max = arr[high];
}
else {
*min = arr[high];
*max = arr[low];
}
return;
}
half

// Divide the array into two halves and recursively find the minimum and maximum of each
mid = (low + high) / 2;
findMinMax(arr, low, mid, &leftMin, &leftMax);
findMinMax(arr, mid+1, high, &rightMin, &rightMax);
// Compare the minimums and maximums of the two halves
if (leftMin < rightMin) {
*min = leftMin;
}
else {
*min = rightMin;
}
if (leftMax > rightMax) {
*max = leftMax;
}
else {
*max = rightMax;
}

}
int main() {
int arr[] = {4, 7, 2, 9, 1, 6, 5, 8, 3};
int n = sizeof(arr) / sizeof(arr[0]);
int min, max;
findMinMax(arr, 0, n-1, &min, &max);
printf("Minimum element is %d\n", min);
printf("Maximum element is %d\n", max);
return 0;
}

// 16.Write a program to find all Hamiltonian Cycles in a connected undirected graph.

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define MAX_VERTICES 100
int graph[MAX_VERTICES][MAX_VERTICES]; // adjacency matrix of the graph
int path[MAX_VERTICES]; // array to store the current path
bool visited[MAX_VERTICES]; // array to keep track of visited vertices
int n; // number of vertices in the graph
// function to print a Hamiltonian cycle
void print_cycle(int path[]) {
printf("Hamiltonian cycle: ");
for (int i = 0; i < n; i++) {
printf("%d ", path[i]);
}
printf("%d\n", path[0]); // print the first vertex again to complete the cycle
}
// function to check if a vertex can be added to the current path
bool is_valid_vertex(int v, int path[], int pos) {
if (graph[path[pos-1]][v] == 0) { // check if there is an edge between the last vertex in
the path and the current vertex
return false;
}
for (int i = 0; i < pos; i++) {
if (path[i] == v) { // check if the current vertex has already been visited in the
path
return false;
}
}
return true;
}
// function to find all Hamiltonian cycles in the graph
void find_hamiltonian_cycles(int path[], int pos) {
if (pos == n) { // all vertices have been visited
if (graph[path[pos-1]][path[0]] == 1) { // check if the last vertex in the path is
adjacent to the first vertex
print_cycle(path);
}
return;
}
for (int i = 1; i < n; i++) { // start from vertex 1 because vertex 0 is already in the
path
if (is_valid_vertex(i, path, pos)) {
path[pos] = i;
visited[i] = true;
find_hamiltonian_cycles(path, pos+1);
visited[i] = false;
}
}
}
int main() {
printf("Enter the number of vertices in the graph (max %d): ", MAX_VERTICES);
scanf("%d", &n);

printf("Enter the adjacency matrix of the graph:\n");
for (int i = 0; i < n; i++) {
for (int j = 0; j < n; j++) {
scanf("%d", &graph[i][j]);
}
}
// initialize the path with vertex 0
path[0] = 0;
visited[0] = true;
// find all Hamiltonian cycles in the graph
find_hamiltonian_cycles(path, 1);
return 0;
}

// 17.Write a program to determine the minimum spanning tree of a graph.

#include <stdio.h>
#include <stdlib.h>
#define MAX_EDGES 1000
#define MAX_VERTICES 100
struct Edge {
int src, dest, weight;
};
struct Graph {
int numVertices, numEdges;
struct Edge edges[MAX_EDGES];
};
struct Subset {
int parent;
int rank;
};
int find(struct Subset subsets[], int i) {
if (subsets[i].parent != i) {
subsets[i].parent = find(subsets, subsets[i].parent);
}
return subsets[i].parent;
}
void unionSet(struct Subset subsets[], int x, int y) {
int xroot = find(subsets, x);
int yroot = find(subsets, y);
if (subsets[xroot].rank < subsets[yroot].rank) {
subsets[xroot].parent = yroot;
} else if (subsets[xroot].rank > subsets[yroot].rank) {
subsets[yroot].parent = xroot;
} else {
subsets[yroot].parent = xroot;
subsets[xroot].rank++;
}
}
int compareEdges(const void* a, const void* b) {
struct Edge* edge1 = (struct Edge*)a;
struct Edge* edge2 = (struct Edge*)b;
return edge1->weight - edge2->weight;
}
void printMST(struct Edge mst[], int numEdges) {
printf("Minimum Spanning Tree:\n");
for (int i = 0; i < numEdges; i++) {
printf("(%d, %d) weight=%d\n", mst[i].src, mst[i].dest, mst[i].weight);
}
}
void kruskal(struct Graph* graph) {
int numVertices = graph->numVertices;
struct Edge mst[numVertices - 1];

qsort(graph->edges, graph->numEdges, sizeof(graph->edges[0]), compareEdges);
struct Subset subsets[numVertices];
for (int i = 0; i < numVertices; i++) {
subsets[i].parent = i;
subsets[i].rank = 0;
}
int numMSTEdges = 0;
int i = 0;
while (numMSTEdges < numVertices - 1) {
struct Edge nextEdge = graph->edges[i++];
int x = find(subsets, nextEdge.src);
int y = find(subsets, nextEdge.dest);
if (x != y) {
mst[numMSTEdges++] = nextEdge;
unionSet(subsets, x, y);
}
}
printMST(mst, numVertices - 1);
}
int main() {
struct Graph graph = { 4, 5, {
{ 0, 1, 10 },
{ 0, 2, 6 },
{ 0, 3, 5 },
{ 1, 3, 15 },
{ 2, 3, 4 }
} };
kruskal(&graph);
return 0;
}

// 18.Write a program to implement 0/1 knapsack using dynamic programming

#include <stdio.h>
#include <stdlib.h>
int max(int a, int b) {
return (a > b) ? a : b;
}
int knapsack(int W, int wt[], int val[], int n) {
int i, w;
int K[n + 1][W + 1];
for (i = 0; i <= n; i++) {
for (w = 0; w <= W; w++) {
if (i == 0 || w == 0)
K[i][w] = 0;
else if (wt[i - 1] <= w)
K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
else
K[i][w] = K[i - 1][w];
}
}
return K[n][W];
}
int main() {
int val[] = {60, 100, 120};
int wt[] = {10, 20, 30};
int W = 50;
int n = sizeof(val) / sizeof(val[0]);
int result = knapsack(W, wt, val, n);
printf("Maximum value: %d\n", result);
return 0;
}

// 19.Write a program to implement N queen's problem using backtracking.

#include <stdio.h>
#include <stdbool.h>
#define N 8

// change this value to set the number of queens

// function to print the solution
void printSolution(int board[N][N]) {
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
printf("%d ", board[i][j]);
}
printf("\n");
}
}
// function to check if a queen can be placed on board[row][col]
bool isSafe(int board[N][N], int row, int col) {
int i, j;
// check this row on left side
for (i = 0; i < col; i++)
if (board[row][i])
return false;
// check upper diagonal on left side
for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
if (board[i][j])
return false;
// check lower diagonal on left side
for (i = row, j = col; j >= 0 && i < N; i++, j--)
if (board[i][j])
return false;
return true;
}
// function to solve N Queen problem using backtracking
bool solveNQueenUtil(int board[N][N], int col) {
if (col == N) {
printSolution(board);
return true;
}
bool res = false;
for (int i = 0; i < N; i++) {
if (isSafe(board, i, col)) {
board[i][col] = 1;
res = solveNQueenUtil(board, col + 1) || res;
board[i][col] = 0; // backtrack
}
}
return res;
}
// function to solve N Queen problem

void solveNQueen() {
int board[N][N] = {0};
if (!solveNQueenUtil(board, 0)) {
printf("Solution does not exist.");
}
}
// main function
int main() {
solveNQueen();
return 0;
}

// 20. Write a program to print all the nodes reachable from a given starting nodes in
digraph using BFS method

#include <stdio.h>
#include <stdlib.h>
#define MAX_NODES 100
// Define a structure for a graph node
typedef struct {
int data;
int visited;
int num_neighbors;
int neighbors[MAX_NODES];
} Node;
// Declare a function for BFS traversal
void BFS(Node* nodes, int start_node);
int main() {
int num_nodes, start_node;
printf("Enter the number of nodes in the graph (maximum %d): ", MAX_NODES);
scanf("%d", &num_nodes);
// Initialize graph nodes
Node nodes[num_nodes];
for (int i = 0; i < num_nodes; i++) {
nodes[i].data = i;
nodes[i].visited = 0;
nodes[i].num_neighbors = 0;
}
// Build the graph by adding edges
int num_edges;
printf("Enter the number of edges: ");
scanf("%d", &num_edges);
printf("Enter the edges (node1 node2): \n");
for (int i = 0; i < num_edges; i++) {
int node1, node2;
scanf("%d %d", &node1, &node2);
nodes[node1].neighbors[nodes[node1].num_neighbors++] = node2;
}
// Prompt user for starting node
printf("Enter the starting node: ");
scanf("%d", &start_node);
// Perform BFS traversal and print reachable nodes
printf("Nodes reachable from node %d: ", start_node);
BFS(nodes, start_node);
printf("\n");
return 0;
}
void BFS(Node* nodes, int start_node) {
// Declare a queue for BFS traversal
int queue[MAX_NODES];
int front = 0, rear = -1;

// Mark the start node as visited and enqueue it
nodes[start_node].visited = 1;
queue[++rear] = start_node;
// Perform BFS traversal
while (front <= rear) {
int current_node = queue[front++];
printf("%d ", current_node);
// Enqueue all unvisited neighbors of current node
for (int i = 0; i < nodes[current_node].num_neighbors; i++) {
int neighbor = nodes[current_node].neighbors[i];
if (!nodes[neighbor].visited) {
nodes[neighbor].visited = 1;
queue[++rear] = neighbor;
}
}
}
}


