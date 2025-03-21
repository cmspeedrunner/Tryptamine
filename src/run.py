import interpreter as pret
import time
import sys
import os
os.system("")
start_time = time.time()  

if not len(sys.argv)-1:
    exit("\033[31mplease supply a file to run\033[m")

file = str(sys.argv[1])
text = "run(\""+file+"\")"
try:
    result, error = pret.run('<stdin>', text)
except KeyboardInterrupt:
    print("\n\033[31m^C\033[0m",end="")
    exit()

if error:
    print("\033[31m" + error.as_string() + "\033[0m")  
elif result:
    if len(result.elements) == 1:
        print(repr(result.elements[0]))
    else:
        print(repr(result))

if "-s" in sys.argv:
    end_time = time.time()  
    elapsed_time = end_time - start_time 
    print('\033[32m' + f"Executed in {elapsed_time:.2f}" + '\033[0m', end="")