import os
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential
import winsound

def text_to_speech_realtime(text: str, voice: str, speed: str = "medium", save_to_file: bool = False):
    # Azure Speech Service Configuration with EntraID authentication
    credential = DefaultAzureCredential()
    
    # Get the access token for Cognitive Services
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    
    # For token-based auth, we need to include the "aad#" prefix and the "#" (hash) separator between resource ID and Microsoft Entra access token to create the access token
    region = os.getenv('AZURE_SPEECH_REGION')
    resourceId = os.getenv('AZURE_SPEECH_RESOURCE_ID')
    auth_token = "aad#" + resourceId + "#" + token.token

    
    # Create speech config with endpoint only, then set the auth token
    # speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SPEECH_KEY"), region=os.getenv("SPEECH_REGION", 'eastus2'))
    speech_config = speechsdk.SpeechConfig(auth_token=auth_token, region=region)
    
    if save_to_file:
        # Create an audio configuration that saves to a file
        audio_filename = "temp_output.wav" 
        audio_output_config = speechsdk.audio.AudioOutputConfig(filename=audio_filename)
    else:
        # Use default speaker for direct output
        audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
    
    # Synthesize speech
    print(f"Synthesizing: '{text}' with voice: {voice}")
    result = speech_synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully!")
        
        if save_to_file:
            print(f"Audio saved to '{audio_filename}'")
            # Play the audio file using Windows winsound module
            try:
                # Use the absolute path for the audio file
                abs_path = os.path.abspath(audio_filename)
                print(f"Playing audio file: {abs_path}")
                winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                print("Audio played successfully!")
            except Exception as e:
                print(f"Failed to play audio: {e}")
                print(f"You can manually play the '{audio_filename}' file to hear the audio")
        else:
            print("Audio played directly to speakers")
            
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")

voice = 'en-US-NancyMultilingualNeural'

# Test with direct speaker output (should be faster)
print("Testing direct speaker output:")
text_to_speech_realtime("Hello, how can I help you?", voice, save_to_file=False)



