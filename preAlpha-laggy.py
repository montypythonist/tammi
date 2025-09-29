from transformers import (
    BeitImageProcessor,
    BeitForImageClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
from PIL import Image
import cv2 as cv
import torch
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
import threading


class VisualCues:
    

    def __init__(self):
        # load processor + model (INITIAL LOADING!!! SHOULD NOT HAPPEN AGAIN AFTER U RUN THIS CODE FOR THE FIRST TIME!)
        self.processor = BeitImageProcessor.from_pretrained(
            "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large"
        )
        self.model = BeitForImageClassification.from_pretrained(
            "Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large"
        )
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

    def predict_probs(self, frame):
        # convert opencv frames to Pillow image to apply models on
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # process and run through model
        inputs = self.processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        return probs  # tensor of probabilities

    def playful_score(self, probs):
        """Heuristic playful/joke/sarcasm detection from visual cues."""
        happy = probs[3].item()
        surprise = probs[6].item()
        # playfulness = happy + surprise basically... idk lmao
        return (happy + 0.5 * surprise)


class AuditoryCues:
    """Audio emotion detection in background."""

    # audio recording settings
    def __init__(self, duration=1, sample_rate=16000):
        self.processor = AutoFeatureExtractor.from_pretrained(
            "Hatman/audio-emotion-detection"
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            "Hatman/audio-emotion-detection"
        )
        self.duration = duration # seconds per recording, 1 by default
        self.sample_rate = sample_rate # required by the model, 16000 by default
        self.latest_probs = torch.zeros(len(self.model.config.id2label))
        self.running = True
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        while self.running:
            """Record audio from microphone and return as numpy array."""
            audio = sd.rec(int(self.duration * self.sample_rate),
                           samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            audio = np.squeeze(audio) # record audio
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt") # preprocess and run model
            with torch.no_grad():
                outputs = self.model(**inputs)
            self.latest_probs = F.softmax(outputs.logits, dim=-1)[0] # softmax to get probabilities

    def playful_score(self):
        """Heuristic playful/joke/sarcasm detection from audio cues."""
        # playfulness = happy + surprise basically... idk lmao (but for audio!)
        # ex: higher playfulness score in audio model indicates playful tone
        # adjust according to Hatman/audio-emotion-detection id2label mapping
        happy_idx = 3 if 3 in self.model.config.id2label else 0
        surprise_idx = 6 if 6 in self.model.config.id2label else 0
        happy = self.latest_probs[happy_idx].item()
        surprise = self.latest_probs[surprise_idx].item()
        return (happy + 0.5 * surprise)


def main():
    visual = VisualCues()
    audio = AuditoryCues(duration=1)
    cam = cv.VideoCapture(0)

    try:
        while True:
            success, frame = cam.read()
            if not success:
                break

            # visual predictions
            visual_probs = visual.predict_probs(frame)
            # get top 3 most likely prediction of emotions
            topk = torch.topk(visual_probs, k=3)
            top_labels = [visual.model.config.id2label[i] for i in topk.indices.tolist()]
            top_probs = topk.values.tolist()
            y0 = 30
            # overlay predictions on video feed
            for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
                text = f"Visual: {label}: {prob*100:.1f}%"
                cv.putText(frame, text, (10, y0 + i*30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # audio predictions (unlike visual predictions, there is only one emotion prediction for audio)
            audio_probs = audio.latest_probs
            audio_top_idx = torch.argmax(audio_probs).item()
            audio_label = audio.model.config.id2label[audio_top_idx]
            audio_conf = audio_probs[audio_top_idx].item() * 100
            audio_text = f"Audio: {audio_label}: {audio_conf:.1f}%"
            # overlay predictions on video feed
            cv.putText(frame, audio_text, (10, y0 + 110), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            # other thing: show all scores for audio for funsies
            # for i, p in enumerate(probs):
            #     print(f"{model.config.id2label[i]:10s}: {p.item():.2%}")

            # multimodal playful/joke/sarcasm detection
            visual_playful = visual.playful_score(visual_probs)
            audio_playful = audio.playful_score()
            combined_playful = (visual_playful + audio_playful) / 2.0
            # overlay predictions on video feed
            if combined_playful > 0.25:  # threshold
                cv.putText(frame, "Likely Joking/Playful/Sarcastic", (10, y0 + 160),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv.imshow("TRUE ADAPTIVE MODELING MULTIEMOTIONAL INTELLIGENCE - Computer Webcam", frame)

            if cv.waitKey(1) & 0xFF == ord("q"): # press "q" on keyboard to end
                break
    finally:
        audio.running = False
        cam.release()
        cv.destroyAllWindows()


# me when i run the code fr
if __name__ == "__main__":
    main()