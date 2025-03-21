#include <iostream>
#include <string>
using namespace std;
int main(int argv, char* argc[]){
    string runCommand = "python src/run.py ";
    if(argv == 1){
        runCommand = "python src/shell.py";
    } else
        for(int i = 1; i < argv; i++){
            runCommand = runCommand+argc[i]+" ";
        }
    system(runCommand.c_str());
    return 0;
}