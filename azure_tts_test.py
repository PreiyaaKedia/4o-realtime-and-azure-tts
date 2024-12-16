import os
import azure.cognitiveservices.speech as speechsdk

def text_to_speech_realtime(text: str, voice: str, speed: str = "medium"):
    # Azure Speech Service Configuration
    speech_config = speechsdk.SpeechConfig(subscription=os.environ['AZURE_SPEECH_KEY'], region=os.environ['AZURE_SPEECH_REGION'])
    audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
    # Synthesize speech
    ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-IN"><voice name="{voice}"><prosody rate="-20%">{text}<mstts:break strength="none"/></prosody></voice></speak>'
    # result = speech_synthesizer.speak_text_async(text).get()
    result = speech_synthesizer.speak_ssml_async(ssml).get()
    for attr in dir(result):
        print(attr)
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully.")
        audio_properties = result.properties
        for attr in dir(audio_properties):
            print(attr)
        # audio_data = result.audio_data
        # # print(audio_data)
        # return audio_data
    else:
        print("Failed to synthesize speech:", result.reason)

voice = 'en-US-NancyMultilingualNeural'
text_to_speech_realtime("Hello, how can I help you?", voice)


