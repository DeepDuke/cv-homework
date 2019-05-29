#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<unordered_map>
#include<set>
using namespace std;

const int maxn = 100;
string sentence = "I am happy and sad";
string substi[3][2] = { {"happy", "glad"}, {"glad", "good"}, {"sad", "sorrow"} };

// build a graph to show substitutes relations
int graph[maxn][maxn];
bool visited[maxn] = { 0 };

// get id for string word
set<string> st;
unordered_map<string, int> str2id;
unordered_map<int, string> id2str;
int cnt = 0;
int getID(string str) {
	if (st.find(str) != st.end())
		return str2id[str];
	int id = cnt++;
	str2id[str] = id;
	id2str[id] = str;
	st.insert(str); // save str
	return id;
}

// used to store substitutes
vector<vector<int> > wordTree;
vector<int> path;

// find all substitutes for a string word
void dfs(int id) {
	if (visited[id])
		return;
	visited[id] = true;
	path.push_back(id);
	//cout << "dfs: " << id2str[id] << endl;
	for (int i = 0; i < cnt; ++i) {
		if (graph[id][i] == 1)
			dfs(i);
	}
	return;
}

// store all possible sentences
vector<int> tempLine;
vector<vector<int> > totalLine;

// generate all possible sentences
void DFS(int id, int depth, int maxDepth) {
	if (depth == maxDepth) { // last floor
		tempLine.push_back(id);
		totalLine.push_back(tempLine);
		tempLine.pop_back();
		return;
	}
	tempLine.push_back(id);
	for (unsigned int i = 0; i < wordTree[depth + 1].size(); ++i) {
		DFS(wordTree[depth + 1][i], depth + 1, maxDepth);
	}
	tempLine.pop_back();
	return;
}

int main()
{
	// build a graph
	for (int i = 0; i < 3; ++i) {
		string s1 = substi[i][0];
		string s2 = substi[i][1];
		int id1 = getID(s1);
		int id2 = getID(s2);
		graph[id1][id2] = 1; // belong to same subgraph
		graph[id2][id1] = 1;
	}
	// split sentence into words
	vector<string> words;
	stringstream ss;
	ss << sentence;
	while (ss.good()) {
		string tempStr;
		ss >> tempStr;
		//cout << tempStr << endl;
		words.push_back(tempStr);
	}
	
	// find substituted partners
	for (auto word : words) {
		path.clear();
		int id = getID(word);
		dfs(id);
		wordTree.push_back(path);
	}
	
	// get all possible substituted sentences
	for (unsigned int i = 0; i < wordTree[0].size(); ++i) {
		DFS(wordTree[0][i], 0, wordTree.size() - 1);
	}
	// output all sentences
	for (unsigned int i = 0; i < totalLine.size(); ++i) {
		string str = "";
		for (unsigned int j = 0; j < totalLine[i].size(); ++j ) {
			int id = totalLine[i][j];
			string tempStr = id2str[id];
			str += tempStr;
			if (j != totalLine[i].size() - 1)
				str += " ";
		}
		cout << str << endl;
	}
	return 0;
}