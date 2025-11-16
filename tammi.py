from tammiDependencies import *
from tammiJSON import *


if visualEnabled:
    
    class VisualCues:

        def __init__(self):
            # load processor + model (INITIAL LOADING!!! SHOULD NOT HAPPEN AGAIN AFTER U RUN THIS CODE FOR THE FIRST TIME!)
            self.processor = BeitImageProcessor.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large") # processor prepares input data for the model
            self.model = BeitForImageClassification.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large") # model is the neural network that makes predictions
            # difference between a processor and a model:
            # processor prepares input data for the model (e.g., normalization, resizing)
            # model is the neural network that makes predictions
            
            # fix label mapping (basically assign each value the emotion name)
            self.model.config.id2label = { # FER-2013 mapping
                0: "anger",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "neutral",
                5: "sad",
                6: "surprise",
            }
            self.prob_history = deque(maxlen=7)  # store last 7 predictions for better smoothing
            self.smoothed_probs = None
            self.latest_emotion = ("neutral", 0.0)

        def predict_probs(self, frame): 
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert opencv frames to Pillow image to apply models on
            pil_img = Image.fromarray(rgb)  # convert numpy array to PIL image
            # what are logits? unnormalized predictions, basically raw scores for each class before applying softmax
            inputs = self.processor(images=pil_img, return_tensors="pt")  # process and run through model
            with torch.no_grad():
                outputs = self.model(**inputs)  # get logits
            temperature = 0.7  # temperature scaling for calibration and confidence adjustment
            logits = outputs.logits / temperature  # apply temperature scaling
            probs = F.softmax(logits, dim=-1)[0]
            self.prob_history.append(probs)  # add to history for smoothing
            smoothed = torch.mean(torch.stack(list(self.prob_history)), dim=0)  # average over last 7 predictions
            self.smoothed_probs = smoothed  # store smoothed probabilities

            # optional: exaggerate or bias non-neutral
            emotion_bias = { # if a specific emotion is not getting detected correctly, change its weight
                # if changing the weight does not help, you can adjust the threshold in tammiMain.py
                "neutral": 0.1, # downplay neutral to make other emotions more prominent
                "happy": 0.9,
                "sad": 1.0,
                "anger": 1.0,
                "fear": 1.2,
                "disgust": 1.3,
                "surprise": 0.7
            }
            adjusted_probs = probs.clone() # copy original probabilities
            for i, label in enumerate(self.model.config.id2label.values()):  # iterate over emotion labels
                bias = emotion_bias.get(label, 1.0)
                adjusted_probs[i] *= bias  # apply bias
            
            # normalize again so probabilities sum to 1
            adjusted_probs /= adjusted_probs.sum()

            # combine smoothed + adjusted probabilities for better recognition of tricky emotions
            exaggerated_probs = (0.6 * smoothed + 0.4 * adjusted_probs)
            exaggerated_probs /= exaggerated_probs.sum()

            # update latest emotion
            top_idx = torch.argmax(exaggerated_probs).item()
            self.latest_emotion = (self.model.config.id2label[top_idx], exaggerated_probs[top_idx].item() * 100)

            return exaggerated_probs  # tensor of probabilities

        if sarcasmEnabled:
            def playful_score(self, probs):
                # Heuristic playful/joke/sarcasm detection from visual cues
                happy = probs[3].item()
                surprise = probs[6].item()
                combined_playfulness = happy + 0.5 * surprise  # playfulness score

                # Optional: override latest_emotion if overly playful/sarcastic
                if self.latest_emotion[0] in ["sad", "anger", "disgust", "surprise"] and combined_playfulness > 0.5:
                    self.latest_emotion = ("playful/sarcastic", combined_playfulness * 100)

                return combined_playfulness

if auditoryEnabled:
    
    class AuditoryCues:

        # audio recording settings
        def __init__(self, duration=0.5, sample_rate=16000):
            # load processor + model (INITIAL LOADING!!! SHOULD NOT HAPPEN AGAIN AFTER U RUN THIS CODE FOR THE FIRST TIME!)
            self.processor = AutoFeatureExtractor.from_pretrained("Hatman/audio-emotion-detection") # processor prepares input data for the model
            self.model = AutoModelForAudioClassification.from_pretrained("Hatman/audio-emotion-detection") # model is the neural network that makes predictions
            self.duration = duration # seconds per recording, 1 by default
            self.sample_rate = sample_rate # MUST BE 16000!!!!!! IF YOU CHANGE IT EVERYTHING BREAKS >:(
            self.latest_probs = torch.zeros(len(self.model.config.id2label)) # initialize with zeros
            self.running = True # flag to control recording thread
            threading.Thread(target=self._record_loop, daemon=True).start() # start recording thread

        def _record_loop(self):
            while self.running:
                audio = sd.rec(int(self.duration * self.sample_rate), # record audio from microphone and return as numpy array
                            samplerate=self.sample_rate, channels=1, dtype="float32")
                sd.wait()
                audio = np.squeeze(audio) # remove single-dimensional entries from the shape of an array
                audio = nr.reduce_noise(y=audio, sr=self.sample_rate) # reduce background noise
                audio = audio / np.max(np.abs(audio) + 1e-8) # normalize volume
                inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt") # preprocess and run model
                with torch.no_grad():
                    outputs = self.model(**inputs) # get logits
                self.latest_probs = F.softmax(outputs.logits, dim=-1)[0] # softmax to get probabilities
                if not hasattr(self, "whisper_model"): # load whisper model only once
                    self.whisper_model = whisper.load_model("tiny") 

                # Transcribe audio
                text_result = self.whisper_model.transcribe(audio, fp16=False) # fp16=False to avoid potential issues on some hardware
                text = text_result["text"]
                self.latest_emotion = ("sarcasm/joking/playful", 100.0) # override latest emotion if sarcasm detected
                
        if sarcasmEnabled:
            def playful_score(self):
                # heuristic playful/joke/sarcasm detection from audio cues
                # playfulness = happy + surprise basically... idk lmao (but for audio!)
                # ex: higher playfulness score in audio model indicates playful tone
                # adjust according to Hatman/audio-emotion-detection id2label mapping
                happy_idx = 3 if 3 in self.model.config.id2label else 0 # fallback to 0 if index not found
                surprise_idx = 6 if 6 in self.model.config.id2label else 0 # fallback to 0 if index not found
                happy = self.latest_probs[happy_idx].item() # get probability of happy emotion
                surprise = self.latest_probs[surprise_idx].item() # get probability of surprise emotion
                return (happy + 0.5 * surprise)
            
# this is line 144 of tammi.py :> isnt that crazy? 144 lines of code in one file, what a madlad