import json

with open('tammiWebcam/tammiConfig.json', 'r') as config:
    # get configuration settings from config.json (what detections does the user want/not want for faster runtime)
    # possible reduction of lag?! this is unprecedented
    config = json.load(config)
    name = config["name"]
    model = config["model"]
    identifier = config["identifier"]
    visualEnabled = config["visualEnabled"]
    auditoryEnabled = config["auditoryEnabled"]
    sarcasmEnabled = config["sarcasmEnabled"]
    
    colorNeutral = config["neutral"]
    colorHappy = config["happy"]
    colorSad = config["sad"]
    colorAnger = config["anger"]
    colorFear = config["fear"]
    colorDisgust = config["disgust"]
    colorSurprise = config["surprise"]
    colorPlayful = config["playful"]
    # inefficient way of cleaning up lag from running processes that doesn't actually work because we haven't implemented it in the regular tammi.py
    # jk we did it now >:3