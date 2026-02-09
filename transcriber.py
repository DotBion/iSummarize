import os
import whisper


print("hello whisper")

class AudioTranscriber:
  def __init__(self, model_size="small"):
    """
    Initializes the transcriber with a specific model size.
    """
    self.model = whisper.load_model(model_size)

    def transcribe_audio(self, audio_path, language="English"):
        """
        Transcribes the audio file specified by audio_path and returns the transcription text.
        """
        result = self.model.transcribe(audio_path, language=language)
        return result["text"]
    
    def save_transcription(self, transcription_text, output_file):
        """
        Saves the transcription text to a .txt file.
        """
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcription_text)

    def transcribe_and_save_from_directory(self, input_dir, output_dir, language="Portuguese"):
        """
        Transcribes all audio files in a specified directory and saves each transcription
        to a separate .txt file in the output directory.

        Args:
            input_dir: Directory containing the audio files to transcribe.
            output_dir: Directory where the transcriptions will be saved.
            language: The language used for transcription (default: "English").
        """
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over all files in the input directory
        for file_name in os.listdir(input_dir):
            # Construct the full file path
            audio_path = os.path.join(input_dir, file_name)

            # Check if the file is an audio file (you can extend this check as needed)
            if os.path.isfile(audio_path) and file_name.lower().endswith(('.mp3', '.mp4', '.wav', '.m4a')):
                # Get the base name of the audio file (without extension)
                base_name = os.path.splitext(file_name)[0]
                output_file = os.path.join(output_dir, f"{base_name}_transcription.txt")

                # Transcribe the audio and save the result
                transcription_text = self.transcribe_audio(audio_path, language)
                self.save_transcription(transcription_text, output_file)
                print(f"Transcription saved to {output_file}")