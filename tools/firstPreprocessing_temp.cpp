#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;

int main(int argc, char **argv) {
    ifstream fin;
    ofstream fout;
    fin.open(argv[1], ios::in);
    if (!fin) {
        cout << "ファイル" << argv[1] << "が開けません" << endl;
        return 1;
    }
    fout.open(argv[2], ios::out|ios::binary|ios::trunc);
    if (!fout) {
        cout << "ファイル" << argv[2] << "が開けません" << endl;
        return 1;
    }
    string line;
    int max_vid = 0;
    while(getline(fin, line)) {
        if (line[0]=='#') continue;
        int src, dst;
        sscanf(line.c_str(), "%d %d", &src, &dst);
        fout.write((char*) &src, sizeof(int));
        fout.write((char*) &dst, sizeof(int));
        if (src > max_vid) max_vid = src;
        if (dst > max_vid) max_vid = dst;
    }
    fin.close();
    fout.close();
    printf("|V|=%d\n", max_vid+1);
    return 0;
}