# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from bark import SAMPLE_RATE, generate_audio, preload_models
import librosa

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        preload_models()

    def predict(
        self,
        text_prompt: str = Input(description="Text prompt"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # generate numpy audio array
        audio_array = generate_audio(text_prompt) 

        # export as wav
        librosa.output.write_wav("/tmp/bark_output.wav", audio_array, SAMPLE_RATE)

        # return path to wav file
        return Path("/tmp/bark_output.wav")
