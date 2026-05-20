import ast

input_file = "input.txt"
output_file = "output.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        # Parse the list safely
        shell_sizes = ast.literal_eval(line)

        # If array is empty, output 0
        if not shell_sizes:
            fout.write("0\n")
        else:
            # First core's vertex count
            fout.write(f"{shell_sizes[0][1]}\n")
