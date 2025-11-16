from tammi import *

def main():
    if visualEnabled:
        visual = VisualCues()
    if auditoryEnabled:
        audio = AuditoryCues(duration=1)
    cam = cv.VideoCapture(0) # CHANGE THIS TO GLASSES INPUT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    while True:
        success, frame = cam.read()
        if not success:
            break

        value1 = "neutral"
        if visualEnabled:
            # visual predictions
            visual_probs = visual.predict_probs(frame)
            # get top 3 most likely prediction of emotions
            topk = torch.topk(visual_probs, k=3)
            top_labels = [visual.model.config.id2label[i] for i in topk.indices.tolist()]
            top_probs = topk.values.tolist()
            # overlay predictions on video feed
            for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
                value1 = label

        value2 = "neutral"
        if auditoryEnabled:
            # audio predictions (unlike visual predictions, there is only one emotion prediction for audio)
            audio_probs = audio.latest_probs
            audio_top_idx = torch.argmax(audio_probs).item()
            audio_label = audio.model.config.id2label[audio_top_idx]
            audio_conf = audio_probs[audio_top_idx].item() * 100
            audio_text = f"Audio: {audio_label}: {audio_conf:.1f}%"
            value2 = audio_label


        # multimodal playful/joke/sarcasm detection
        if sarcasmEnabled:
            if visualEnabled:
                visual_playful = visual.playful_score(visual_probs)
            if auditoryEnabled:
                audio_playful = audio.playful_score()
            if visualEnabled and auditoryEnabled:
                combined_playful = (visual_playful + audio_playful) / 2.0
            elif visualEnabled and not auditoryEnabled:
                combined_playful = visual_playful
            elif auditoryEnabled and not visualEnabled:
                combined_playful = audio_playful
            # overlay predictions on video feed
            if combined_playful > 0.25:  # threshold
                value1, value2 = "playful"
        
        # whats gaming my gamers
        emotionColor = {
            "neutral": colorNeutral,
            "happy": colorHappy,
            "sad": colorSad,
            "angry": colorAngry,
            "surprise": colorSurprise,
            "fear": colorFear,
            "disgust": colorDisgust,
            "playful": colorPlayful,
        }
        for i in emotionColor.keys():
            if value1 == i:
                color = emotionColor[i]
            if value2 == i:
                color2 = emotionColor[i]
            
        # write code here for putting the colors onto the glasses
        
            
# me when i run the main program and all my functions get called, frfr
if __name__ == "__main__":
    main()