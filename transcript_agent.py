import librosa
import noisereduce as nr
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import os


# whisper_model = WhisperModel("/ssd1/yujunqiu/model_zoo/Systran/faster-whisper-large-v2", device="cuda")


# audio_path = "/tmp/tmpjol9m22n.wav"
# result = whisper_model.transcribe(audio_path)


class AudioProcessingAgent:
    def __init__(self,
                 vad_model: str = "pyannote/voice-activity-detection",
                 diarization_model: str = "pyannote/speaker-diarization",
                 whisper_model: str = "/home/dell/mwy/faster-whisper-large-v2",
                 whisper_device: str = "cuda"):
        self.vad_pipeline = Pipeline.from_pretrained(vad_model)
        self.diarization_pipeline = Pipeline.from_pretrained(diarization_model)
        self.whisper_model = WhisperModel(whisper_model, device=whisper_device)

    def preprocess_audio(self, input_path: str, target_sr: int = 16000) -> str:
        y, sr = librosa.load(input_path, sr=target_sr)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        temp_wav = tempfile.mktemp(suffix=".wav")
        audio = AudioSegment(
            y_denoised.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        audio.export(temp_wav, format="wav")
        return temp_wav

    def segment_audio(self, audio_path: str, merge_threshold: float = 0.5) -> list:
        vad_results = self.vad_pipeline(audio_path)
        segments = []
        current_start = None
        for speech in vad_results.get_timeline().support():
            if current_start is None:
                current_start = speech.start
                current_end = speech.end
            elif speech.start - current_end < merge_threshold:
                current_end = speech.end
            else:
                segments.append((current_start, current_end))
                current_start = speech.start
                current_end = speech.end
        if current_start is not None:
            segments.append((current_start, current_end))
        return segments

    def detect_language(self, audio_path: str) -> tuple:
        _, info = self.whisper_model.transcribe(audio_path, beam_size=5)
        return info.language, info.language_probability

    def speaker_diarization(self, audio_path: str) -> list:
        diarization = self.diarization_pipeline(audio_path)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return speaker_segments

    def transcribe_segments(self, audio_path: str, segments: list) -> list:
        full_audio = AudioSegment.from_wav(audio_path)
        results = []
        for idx, (start, end) in enumerate(segments):
            chunk = full_audio[start * 1000:end * 1000]
            temp_chunk = tempfile.mktemp(suffix=".wav")
            chunk.export(temp_chunk, format="wav")
            segment_text = ""
            for seg, _ in self.whisper_model.transcribe(temp_chunk):
                segment_text += seg.text + " "
            results.append({
                "start": start,
                "end": end,
                "text": segment_text.strip()
            })
        return results

    def process_audio(self, input_path: str) -> dict:
        """
        整体流程：预处理 → 分段 → 转录 → 说话人 → 语言识别
        """
        print("Step 1: 预处理音频...")
        clean_wav = self.preprocess_audio(input_path)

        print("Step 2: 语音活动检测 + 动态分片...")
        segments = self.segment_audio(clean_wav)

        print("Step 3: 分段转录...")
        transcription = self.transcribe_segments(clean_wav, segments)

        print("Step 4: 说话人识别...")
        speakers = self.speaker_diarization(clean_wav)

        print("Step 5: 语言检测...")
        language, prob = self.detect_language(clean_wav)

        return {
            "segments": segments,
            "transcription": transcription,
            "speaker_diarization": speakers,
            "language": language,
            "language_confidence": prob
        }


if __name__ == "__main__":
    agent = AudioProcessingAgent()

    result = agent.process_audio("./data/huggingcast-s2e6.m4a")
    print(result)