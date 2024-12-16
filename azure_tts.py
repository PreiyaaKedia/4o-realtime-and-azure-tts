# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import asyncio
import logging
import os
from typing import AsyncIterator, Tuple
import traceback
import azure.cognitiveservices.speech as speechsdk
import numpy as np
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

async def text_to_speech_realtime(speech_synthesizer : speechsdk.SpeechSynthesizer, text: str, voice: str, speed: str = "medium"):
    # Azure Speech Service Configuration
    speech_config = speechsdk.SpeechConfig(subscription=os.environ['AZURE_SPEECH_KEY'], region=os.environ['AZURE_SPEECH_REGION'])
    # Synthesize speech
    ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-IN"><voice name="{voice}"><prosody rate="+10%">{text}</prosody></voice></speak>'
    # result = speech_synthesizer.speak_text_async(text).get()
    result = speech_synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully.")
        audio_data = result.audio_data
        # print(audio_data)
        return audio_data
    else:
        print("Failed to synthesize speech:", result.reason)