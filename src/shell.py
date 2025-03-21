import interpreter
import sys, os
os.system("")
header = f"""\u001B[36m
╔╦╗╦═╗╦ ╦╔═╗╔╦╗╔═╗╔╦╗╦╔╗╔╔═╗
 ║ ╠╦╝╚╦╝╠═╝ ║ ╠═╣║║║║║║║║╣ 
 ╩ ╩╚═ ╩ ╩   ╩ ╩ ╩╩ ╩╩╝╚╝╚═╝\u001B[35m
╔────────────────────────╗  
│\u001B[32mV{interpreter.VERSION}\u001B[35m                   │  
│\u001B[32mMade by CM\u001B[35m              │  
│with credit to \u001B[32mangelcaru\u001B[35m│  
╚────────────────────────╝  """

def main():
	print(header)
	if len(sys.argv) > 1:
		dyr, fn = os.path.split(sys.argv[1])
		os.chdir(dyr)
		with open(fn, "r") as f:
			code = f.read()
		_, error = interpreter.run(fn, code)
		if error:
			print(error.as_string(), file=sys.stderr)
			exit(1)
		exit()
	
	while True:
		text = input('\u001B[36mTryp>\033[0m')
		if text.strip() == "": continue

		else:

			result, error = interpreter.run('<stdin>', text)

		if error:
			print(error.as_string(), file=sys.stderr)
		elif result:
			real_result = result.elements[0]
			if len(result.elements) != 1:
				real_result = result
			print(repr(real_result))
			interpreter.global_symbol_table.set("_", real_result)

if __name__ == "__main__":
	main()
