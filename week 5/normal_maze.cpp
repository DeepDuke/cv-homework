#include<iostream>
#include<vector>
#include<utility>
#include<algorithm>
using namespace std;

const int N = 5;
const int INF = 88888888;

vector<pair<int, int> > path, bestPath; 

bool visited[N][N] = {0};
int maze[N][N] = {
					{1, 5, 5, 5, 1},
					{1, 2, 2, 2, 1},
					{1, 1, 1, 1, 1},
					{3, 4, 4, 4, 0},
					{3, 3, 3, 3, 1}
				};
void dfs(pair<int, int> now, pair<int, int> end, int nowCost, int &minCost){
	if(now.first < 0 || now.first >= N || now.second < 0 || now.second >= N) // exceed limitation
		return;
	if(visited[now.first][now.second]) // has been visited before
		return;
	if(now == end){
		//reach end
		path.push_back(now);
		nowCost += maze[now.first][now.second];
		// compare total cost with minCost
		if(nowCost < minCost){
			minCost = nowCost;
			bestPath = path;
		}
		path.pop_back();
		return;
	}
	path.push_back(now);
	visited[now.first][now.second] = true;
	
	pair<int, int> next;
	// left
	next.first = now.first;
	next.second = now.second - 1;
	dfs(next, end, nowCost+maze[now.first][now.second], minCost);

	// right
	next.first = now.first;
	next.second = now.second + 1;
	dfs(next, end, nowCost+maze[now.first][now.second], minCost);

	// up
	next.first = now.first - 1;
	next.second = now.second;
	dfs(next, end, nowCost+maze[now.first][now.second], minCost);

	// down
	next.first = now.first + 1;
	next.second = now.second;
	dfs(next, end, nowCost+maze[now.first][now.second], minCost);
	
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
		cout << "(" << bestPath[i].first << "," << bestPath[i].second << ")" << endl;
	}
	cout << endl;
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
	pair<int, int> s1 (0, 0);
	pair<int, int> t1 (4, 4);
	int minCost = INF;
	dfs(s1, t1, 0, minCost);
	cout << "test1:\n" << "minimal cost is : " << minCost << endl;
	myPrint();
	
	// test 2
	path.clear();
	bestPath.clear();
	minCost = INF;
	pair<int, int> s2 (0, 0);
	pair<int, int> t2 (0, 4);
	dfs(s2, t2, 0, minCost);
	cout << "test2:\n" << "minimal cost is : " << minCost << endl;
	myPrint();
	return 0;
}
