import os

file = open("diffs.txt", "r", encoding="utf-8")

mapping = dict()
for line in file:
	line = line.replace(".py", "")
	line = line.replace("/", ".")
	line = line[:-4].strip()
	leftsplit = line.split("{", 1)
	rightsplit = leftsplit[1].split("}", 1)
	left = leftsplit[0]
	diff = rightsplit[0].split(" => ")
	right = rightsplit[1]
	old = (left+diff[0]+right).replace("..", ".")
	new = (left+diff[1]+right).replace("..", ".")
	mapping[old] = new
	print(left, diff, right, sep=", ")
file.close()
print(mapping)

for x in os.walk("."):
	if ".git" in x[0] or "__pycache__" in x[0]:
		continue
	for filename in x[2]:
		if filename[-3:] != ".py" or filename == "fixRefs.py":
			continue
		filename = x[0] + "/" + filename
		print("\n", filename)
		output = ""
		file = open(filename, "r", encoding="utf-8")
		for line in file:
			if line[:4] == "from" and "import" in line:
				source = line[5:].split(" import", 1)[0]
				if source in mapping:
					print(source, "=>", mapping[source])
					line = line.replace(source, mapping[source])
			output += line
		file.close()
		if output[-1] == "\n":
			output = output[:-1]
		
		file = open(filename, "w", encoding="utf-8")
		print(output, file=file)
		file.close()

# git log --stat -n 1 | grep "=>" | grep "| 0" > diffs.txt
