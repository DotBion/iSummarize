import os
import openai
from dotenv import load_dotenv

class AudioSummaryGenerator:
    def __init__(self):
        """
        Initializes the summary generator and loads the OpenAI API key from .env.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Get the OpenAI API key from the environment variable
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("OpenAI API key not found. Please set it in the .env file.")

        # Set the API key for OpenAI
        openai.api_key = openai_api_key

    def generate_summary(self, transcription, report_structure=""):
        """
        Generates a summary based on the transcription and the provided report structure using OpenAI's GPT model.
        """
        prompt = f"""
        Here is the transcription of an email:
        {transcription}

        Based on the transcription, fill in the fields below in the report format
        """

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access to it
            messages=[
                {"role": "system", "content": "You are an assistant that generates reports from audio transcriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )

        # Extract the response text
        summary = response.choices[0].message.content
        print(summary)
        return summary
    
c = AudioSummaryGenerator()
c.generate_summary("The rapid evolution of artificial intelligence has revolutionized numerous industries, from healthcare and finance to entertainment and autonomous systems. Deep learning frameworks like PyTorch have made it easier for researchers and developers to experiment with complex neural networks, enabling breakthroughs in image recognition, natural language processing, and robotics. With its dynamic computation graph and intuitive Pythonic interface, PyTorch has become a preferred choice for both academic research and industrial applications. Beyond AI, technological advancements in robotics and automation are reshaping the workforce, enhancing efficiency while also raising ethical concerns about job displacement and data privacy. As we move toward an increasingly interconnected world, the integration of AI with Internet of Things (IoT) devices is creating smarter environments, from self-regulating traffic systems to personalized healthcare solutions. These innovations underscore the need for continuous learning and adaptation, ensuring that humanity harnesses technology for progress while addressing its challenges responsibly.")