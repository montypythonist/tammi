import json

with open('tammiWebcam/tammiConfig.json', 'r') as config:
    # get configuration settings from config.json (what detections does the user want/not want for faster runtime)
    # possible reduction of lag?! this is unprecedented
    config = json.load(config)
    visualEnabled = config["visualEnabled"]
    auditoryEnabled = config["auditoryEnabled"]
    sarcasmEnabled = config["sarcasmEnabled"]
    # inefficient way of cleaning up lag from running processes that doesn't actually work because we haven't implemented it in the regular tammi.py
    # jk we did it now >:3