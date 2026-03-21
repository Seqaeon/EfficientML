import json

with open("kv_cache_experiments.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if '"thinkk_num_ctx":' in source:
            source = source.replace('"thinkk_num_ctx"', '"thinkkv_num_ctx"')
            source = source.replace('"thinkk_temperature"', '"thinkkv_temperature"')
            cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]
            if cell["source"][-1] == "":
                cell["source"] = cell["source"][:-1]
    elif cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "`thinkk`" in source:
            source = source.replace("`thinkk`", "`thinkkv`")
            cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]
            if cell["source"][-1] == "":
                cell["source"] = cell["source"][:-1]

with open("kv_cache_experiments.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Fixed thinkk_num_ctx and markdown")
