from tammiDependencies import *
from tammiJSON import *


if visualEnabled:
    
    class VisualCues:
            

        def __init__(self):
            # load processor + model (INITIAL LOADING!!! SHOULD NOT HAPPEN AGAIN AFTER U RUN THIS CODE FOR THE FIRST TIME!)
            self.processor = BeitImageProcessor.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large")
            self.model = BeitForImageClassification.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large")
            # fix label mapping (basically assign each value the emotion name)
            self.model.config.id2label = {
                0: "anger",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "neutral",
                5: "sad",
                6: "surprise",
            }

        def predict_probs(self, frame): # TODO: test this with more than one face please? if users are in a group convo they will get damned to hell if this isn't implemented
            # convert opencv frames to Pillow image to apply models on
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # process and run through model
            inputs = self.processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            return probs  # tensor of probabilities
        
        if sarcasmEnabled:
            def playful_score(self, probs):
                # Heuristic playful/joke/sarcasm detection from visual cues
                happy = probs[3].item()
                surprise = probs[6].item()
                # playfulness = happy + surprise basically... idk lmao
                return (happy + 0.5 * surprise)


if auditoryEnabled:
    
    class AuditoryCues:

        # audio recording settings
        def __init__(self, duration=1, sample_rate=16000):
            self.processor = AutoFeatureExtractor.from_pretrained("Hatman/audio-emotion-detection")
            self.model = AutoModelForAudioClassification.from_pretrained("Hatman/audio-emotion-detection")
            self.duration = duration # seconds per recording, 1 by default
            self.sample_rate = sample_rate # required by the model, 16000 by default
            self.latest_probs = torch.zeros(len(self.model.config.id2label))
            self.running = True
            threading.Thread(target=self._record_loop, daemon=True).start()

        def _record_loop(self):
            while self.running:
                # Record audio from microphone and return as numpy array
                audio = sd.rec(int(self.duration * self.sample_rate),
                            samplerate=self.sample_rate, channels=1, dtype="float32")
                sd.wait()
                audio = np.squeeze(audio) # record audio
                inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt") # preprocess and run model
                with torch.no_grad():
                    outputs = self.model(**inputs)
                self.latest_probs = F.softmax(outputs.logits, dim=-1)[0] # softmax to get probabilities
                
        if sarcasmEnabled:
            def playful_score(self):
                # Heuristic playful/joke/sarcasm detection from audio cues
                # playfulness = happy + surprise basically... idk lmao (but for audio!)
                # ex: higher playfulness score in audio model indicates playful tone
                # adjust according to Hatman/audio-emotion-detection id2label mapping
                happy_idx = 3 if 3 in self.model.config.id2label else 0
                surprise_idx = 6 if 6 in self.model.config.id2label else 0
                happy = self.latest_probs[happy_idx].item()
                surprise = self.latest_probs[surprise_idx].item()
                return (happy + 0.5 * surprise)