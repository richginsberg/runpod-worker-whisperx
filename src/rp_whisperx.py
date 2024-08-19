import whisperx
import gc
import torch
import os
import json
import config

device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)
ALLOWED_EXTENSIONS = {"mp3", "wav", "awb", "aac", "ogg", "oga", "m4a", "wma", "amr"}

class WhisperX_worker:

    def process(self, audio, pretty_json):
        audio_file = str(audio)
        if self.allowed_file(audio_file):            
            print(audio)
            result = self.transcribe(audio)

            if pretty_json is True:
                return json.dumps(result, indent=4)
            else:
                return json.dumps(result)

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def transcribe(audio_file):
        # Load models
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=config.HF_TOKEN, device=device)
        #global model, diarize_model
        # 1. Transcribe with original whisper (batched)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # 3. Assign speaker labels
        diarize_segments = diarize_model(audio_file)

        # Return result
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del model, diarize_model
        gc.collect
        torch.cuda.empty_cache()
        return result