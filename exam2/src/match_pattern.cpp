// k8r@uw.edu
// final exam simple starter code
// compiling:
// g++ -std=c++14 -c -I./ match_pattern.cpp ; g++ -std=c++14 -o xmatch_pattern match_pattern.o 
// running:
// ./xmatch_pattern 

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

#include "match_pattern.hpp"

int main(int argc, char** argv) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " <text_file>" << endl;
        return 1;
    }
    string text_file = string(argv[1]);
    cout << "Reading text from file: " << text_file << endl;

    // Read the file content
    ifstream inputFile(text_file);
    if (!inputFile) {
        cerr << "Error: Unable to open input file." << endl;
        return 1;
    }
    string text((istreambuf_iterator<char>(inputFile)), istreambuf_iterator<char>());
    inputFile.close();

    // Get the pattern to match
    const std::string alphabet = "ABCDEFGHIJKLMN";
    string pattern = "";
    while(1)
    {
        cout << "Enter a string (length <= 7) from alphabet {" << alphabet << "} to match: (or 'exit') ";
        cin >> pattern;
        if(pattern == "exit") { break; }

        if (pattern.length() > 7) {
            cerr << "Error: Pattern length must be 7 or less." << endl;
            return 1;
        }

        // call your pattern matching algorithm
        matchPattern(text, pattern);
    }

    return 0;
}

