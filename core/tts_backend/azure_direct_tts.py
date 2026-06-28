import requests
from core.utils import load_key

def azure_direct_tts(text: str, save_path: str) -> None:
    region = load_key("azure_direct_tts.region")
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    api_key = load_key("azure_direct_tts.api_key")
    voice = load_key("azure_direct_tts.voice")

    payload = f"<speak version='1.0' xml:lang='en-US'><voice name='{voice}'>{text}</voice></speak>"
    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
        'Content-Type': 'application/ssml+xml',
        'User-Agent': 'VideoLingo'
    }

    response = requests.post(url, headers=headers, data=payload.encode('utf-8'))
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Audio saved to {save_path}")

if __name__ == "__main__":
    azure_direct_tts("Hi! Welcome to VideoLingo!", "test.wav")
