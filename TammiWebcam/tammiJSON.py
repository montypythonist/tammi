import json

with open('tammiWebcam/tammiConfig.json', 'r') as config:
    # get configuration settings from config.json (what detections does the user want/not want for faster runtime)
    config = json.load(config)
    name = config["name"]
    model = config["model"]
    identifier = config["identifier"]
    visualEnabled = config["visualEnabled"]
    auditoryEnabled = config["auditoryEnabled"]
    sarcasmEnabled = config["sarcasmEnabled"]
    # inefficient way of cleaning up lag from running processes that doesn't actually work because we haven't implemented it in the regular tammi.py