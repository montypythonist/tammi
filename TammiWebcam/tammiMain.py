from tammi import *

def main():
    if visualEnabled:
        visual = VisualCues()
    if auditoryEnabled:
        audio = AuditoryCues(duration=1)
    cam = cv.VideoCapture(0)

    try:
        while True:
            success, frame = cam.read()
            if not success:
                break

            if visualEnabled:
                # visual predictions
                visual_probs = visual.predict_probs(frame)
                # get top 3 most likely prediction of emotions
                topk = torch.topk(visual_probs, k=3)
                top_labels = [visual.model.config.id2label[i] for i in topk.indices.tolist()]
                top_probs = topk.values.tolist()
                # overlay predictions on video feed
                for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
                    text = f"Visual: {label}: {prob*100:.1f}%"
                    cv.putText(frame, text, (10, 30 + i*30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if auditoryEnabled:
                # audio predictions (unlike visual predictions, there is only one emotion prediction for audio)
                audio_probs = audio.latest_probs
                audio_top_idx = torch.argmax(audio_probs).item()
                audio_label = audio.model.config.id2label[audio_top_idx]
                audio_conf = audio_probs[audio_top_idx].item() * 100
                audio_text = f"Audio: {audio_label}: {audio_conf:.1f}%"
                # overlay predictions on video feed... shocker.
                # this only presents one prediction and the below code doesn't actually implement correctly. too bad!
                
                cv.putText(frame, audio_text, (10, 30 + 110), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                # other thing: show all scores for audio for funsies
                # for i, p in enumerate(probs):
                #     print(f"{model.config.id2label[i]:10s}: {p.item():.2%}")


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
                    cv.putText(frame, "Likely Joking/Playful/Sarcastic", (10, 30 + 160),
                            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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