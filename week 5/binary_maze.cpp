#include<iostream>
#include<vector>
#include<utility> 
#include<algorithm>
using namespace std; 

const int N = 5;
const int INF = 88888888;
// N*N maze: 1 means wall, 0 means the empty space
int maze[N][N] = {
					{0, 0, 1, 0, 0},
					{0, 0, 0, 0, 0},
					{0, 0, 0, 1, 0},
					{1, 1, 0, 1, 1},
					{0, 1, 0, 0, 0}
				};
bool visited[N][N] = {0};

vector<pair<int, int> > bestPath;

void dfs(pair<int, int> now, pair<int, int> end, int depth, int &minDepth, vector<pair<int, int> > &path){
	if(now.first < 0 || now.first >= N || now.second < 0 || now.second >= N) // exceed limitation
		return;
	if(visited[now.first][now.second]) // has been visited before
		return;
	if(maze[now.first][now.second] == 1) // wall
		return; 
	if(now == end) {
		path.push_back(end);
		if(depth < minDepth){
			bestPath = path;
			minDepth = depth;
		}
		path.pop_back();		
		return; // find destination
	}
	// output now
	//cout << "depth: " << depth << " now: " << "(" << now.first << "," << now.second << ")" << endl;
	path.push_back(now);
	visited[now.first][now.second] = true;
	
	pair<int, int> next;
	// left
	next.first = now.first;
	next.second = now.second - 1;
	dfs(next, end, depth+1, minDepth, path);

	// right
	next.first = now.first;
	next.second = now.second + 1;
	dfs(next, end, depth+1, minDepth, path);

	// up
	next.first = now.first - 1;
	next.second = now.second;
	dfs(next, end, depth+1, minDepth, path);

	// down
	next.first = now.first + 1;
	next.second = now.second;
	dfs(next, end, depth+1, minDepth, path);
	
	path.pop_back();
	visited[now.first][now.second] = false;
	
	return;
	
}
void myPrint(){
	// output shortest path
	cout << "path length: " << bestPath.size() << endl;
	if(bestPath.size() == 0){
		cout << "There is no way from start to end!" << endl;
		return;
	}
	for(unsigned int i = 0; i < bestPath.size(); ++i){
		cout << "(" << bestPath[i].first << "," << bestPath[i].second << ")";
		if (i != bestPath.size() - 1)
			cout << " -> ";
	}
	cout << endl << endl;
	return;
}
int main()
{
	//output maze
	cout << "The maze is :\n\n";
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			cout << maze[i][j];
			if (j != N - 1)
				cout << " ";
		}
		cout << endl;
	}
	cout << endl;
	// test 1
	pair<int, int> s1 (0, 4);
	pair<int, int> t1 (4, 4);
    int minDepth = INF;
    vector<pair<int, int> > path;
	dfs(s1, t1, 1, minDepth, path);
	cout << "test1: from (0, 4) to (4, 4)\n";
	myPrint();
	
	//test 2
	fill(visited[0], visited[0]+N*N, false);
	bestPath.clear();
	path.clear();
	minDepth = INF;
	pair<int, int> s2 (0, 4);
	pair<int, int> t2 (3, 2);
	dfs(s2, t2, 1, minDepth, path);
	cout << "test2: from (0, 4) to (3, 2)\n";
	myPrint();
	
	// test 3
	fill(visited[0], visited[0]+N*N, false);
	bestPath.clear();
	path.clear();
	minDepth = INF;
	pair<int, int> s3 (0, 4);
	pair<int, int> t3 (4, 0);
	dfs(s3, t3, 1, minDepth, path);
	cout << "test3: from (0, 4) to (4, 0)\n";
	myPrint();
	return 0;
}
