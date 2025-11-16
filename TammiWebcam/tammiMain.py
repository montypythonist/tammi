from tammi import *


def main():
    if visualEnabled: # if the user enabled visual emotion detection
        visual = VisualCues() 
    if auditoryEnabled: # if the user enabled auditory emotion detection
        audio = AuditoryCues(duration=1)
    cam = cv.VideoCapture(0)

    try:
        while True:
            success, frame = cam.read() # cv2 dont fail me now!!!!
            if not success:
                break
            
            value1, value2 = ""
            if visualEnabled:
                visual_probs = visual.predict_probs(frame) # visual predictions
                topk = torch.topk(visual_probs, k=1) # get most likely prediction of emotions
                top_label = [visual.model.config.id2label[i] for i in topk.indices.tolist()] # map indices to label
                top_prob = topk.values.tolist() # get probabilities
                
                # overlay predictions on video feed
                for i, (label, prob) in enumerate(zip(top_label, top_prob)):
                    value1 = str(label)
                    # text = f"Visual: {label}: {prob*100:.1f}%"
                    # cv.putText(frame, text, (10, 30 + i*30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    

            if auditoryEnabled:
                # audio predictions (unlike visual predictions, there is only one emotion prediction for audio)
                audio_probs = audio.latest_probs # get latest audio probabilities
                audio_top_idx = torch.argmax(audio_probs).item() # index of most likely emotion
                audio_label = audio.model.config.id2label[audio_top_idx] # map index to label
                # audio_conf = audio_probs[audio_top_idx].item() * 100 # confidence of most likely emotion
                value2 = str(audio_label)
                # audio_text = f"Audio: {audio_label}: {audio_conf:.1f}%" 
                # overlay predictions on video feed... shocker.
                # cv.putText(frame, audio_text, (10, 30 + 110), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


            # multimodal playful/joke/sarcasm detection
            # existing visual_probs from predict_probs()
            visual_probs = visual.predict_probs(frame) if visualEnabled else None

            # sarcasm/playfulness detection
            if sarcasmEnabled:  # if the user enabled sarcasm detection
                # warning: peak jank performance below >:/ 
                # the complicated nest of if statements is to handle all 3 cases of visual/audio/both being enabled
                # if only one is enabled, just use that one's playful score
                if visualEnabled:
                    visual_playful = visual.playful_score(visual_probs)
                if auditoryEnabled:
                    audio_playful = audio.playful_score()
                if visualEnabled and auditoryEnabled:
                    combined_playful = (visual_playful + audio_playful) / 2.0  # average the two scores
                elif visualEnabled and not auditoryEnabled:
                    combined_playful = visual_playful
                elif auditoryEnabled and not visualEnabled:
                    combined_playful = audio_playful
                else:
                    combined_playful = 0.0  # fallback if neither enabled

                # overlay predictions on video feed (jk)
                if combined_playful > 0.25:  # threshold
                    value1, value2 = "playful"

            # whats gaming my gamers
            colorAll = [colorNeutral, colorHappy, colorSad, colorAnger, colorFear, colorSurprise, colorDisgust, colorPlayful]
            emotionAll = ["neutral", "happy", "sad", "anger", "fear", "surprise", "disgust", "playful"]
            for i in emotionAll:
            # once we get hardware this will make sense
                if value1 == i:
                    visualValue = colorAll[i]
                if value2 == i:
                    audioValue = colorAll[i]
            
            cv.imshow("TRUE ADAPTIVE MODELING MULTIEMOTIONAL INTELLIGENCE - Computer Webcam", frame)

            if cv.waitKey(1) & 0xFF == ord("q"): # press "q" on keyboard to end
                break
    finally: # end detections here if user ends the program
        if auditoryEnabled:
            audio.running = False
        cam.release()
        cv.destroyAllWindows()


# me when i run the main program and all my functions get called, frfr
if __name__ == "__main__":
    main()