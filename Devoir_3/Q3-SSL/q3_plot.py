import json
import torch
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot

loggers = dict()
lst_experiment = ["sgT", "sgF", "prN", "dim4096"]
for expriment in lst_experiment:
    with open(f"logs/pretrain/log_{expriment}" + ".json", 'r') as f:
        loggers["pretrain_" + expriment] = json.load(f)

for expriment in lst_experiment:
    with open(f"logs/classification/log_{expriment}" + ".json", 'r') as f:
        loggers["classification_" + expriment] = json.load(f)


matplotlib.pyplot.figure("sg")
matplotlib.pyplot.plot(loggers["classification_sgT"]["t_accu"], label="w stop-gradient / train")
matplotlib.pyplot.plot(loggers["classification_sgF"]["t_accu"], label="w/o stop-gradient / train")

matplotlib.pyplot.plot(loggers["classification_sgT"]["v_accu"], label="w stop-gradient / valid")
matplotlib.pyplot.plot(loggers["classification_sgF"]["v_accu"], label="w/o stop-gradient / valid")

matplotlib.pyplot.xlabel("training epoch")
matplotlib.pyplot.ylabel("accuracy")
matplotlib.pyplot.legend()
matplotlib.pyplot.savefig("results/p3q4_class_accu.pdf")
matplotlib.pyplot.show()







