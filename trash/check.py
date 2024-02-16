import os
Datadirectory = "processed_ds/"
c = 0
for i in os.listdir(Datadirectory):
    print(len(os.listdir("processed_ds/"+i)))