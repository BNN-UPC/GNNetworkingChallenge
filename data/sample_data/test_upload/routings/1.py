import os

files = [f for f in os.listdir("./") if f.endswith("txt")]
for f in files:
  os.system("cp "+f+" 1/Routing-rediris"+f[7:])
