#ifndef MATCH_PATTERN_HPP
#define MATCH_PATTERN_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

void matchPattern(const string& text, const string& pattern) {
    int m = pattern.size();
    int n = text.size();
    vector<int> occurrences;

    // execute your pattern matching
    for(auto i = 0; i < n; ++i)
    {
        bool match = true;
        for(auto j = 0; j < m; ++j)
        {
            if(text[i+j] != pattern[j])
            {
                match = false;
                break;
            }
        }
        if(match) { occurrences.push_back(i); }   
    }

    // print the results
    cout << "Pattern found " << occurrences.size() << " times at positions: ";
    for (int pos : occurrences) {
        cout << pos << " ";
    }
    cout << endl;
}

#endif // MATCH_PATTERN_HPP

