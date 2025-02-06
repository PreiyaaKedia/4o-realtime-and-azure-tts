import os
import traceback
import asyncio
import azure.cognitiveservices.speech as speechsdk
from openai import AsyncAzureOpenAI
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from uuid import uuid4
from chainlit.logger import logger
from realtime import RealtimeClient
from azure_tts import text_to_speech_realtime
from dotenv import load_dotenv
from tools import tools

voice = "en-IN-AnanyaNeural"

load_dotenv()
client_contexts = {}

VOICE_MAPPING = {
    "english": "en-US-NancyMultilingualNeural",
    "hindi": "hi-IN-SwaraNeural",
    "tamil": "ta-IN-NancyMultilingualNeural",
    "odia": "or-IN-NancyMultilingualNeural",
    "bengali": "bn-IN-NancyMultilingualNeural",
    "gujarati": "gu-IN-NancyMultilingualNeural",
    "kannada": "kn-IN-NancyMultilingualNeural",
    "malayalam": "ml-IN-NancyMultilingualNeural",
    "marathi": "mr-IN-NancyMultilingualNeural",
    "punjabi": "pa-IN-NancyMultilingualNeural",
    "telugu": "te-IN-NancyMultilingualNeural",
    "urdu": "ur-IN-NancyMultilingualNeural"
}

tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n", "।"]
interrupt_audio = False
collected_messages = []
item_id = None

async def setup_openai_realtime(system_prompt: str):
    global client_contexts
    global collected_messages
    global item_id
    """Instantiate and configure the OpenAI Realtime Client"""
    openai_realtime = RealtimeClient(system_prompt = system_prompt)
    speech_config = speechsdk.SpeechConfig(subscription=os.environ['AZURE_SPEECH_KEY'], endpoint=f'wss://{os.environ['AZURE_SPEECH_REGION']}.tts.speech.microsoft.com/cognitiveservices/websocket/v1?enableTalkingAvatar=false')
    # speech_config = speechsdk.SpeechConfig(subscription=os.environ['AZURE_SPEECH_KEY'], region=os.environ['AZURE_SPEECH_REGION'])
    # speech_config.speech_synthesis_voice_name = VOICE_MAPPING.get(cl.user_session.get("Language"), "en-US-NancyMultilingualNeural")
    speech_config.speech_synthesis_voice_name = "en-US-NancyMultilingualNeural"
    # logger.info("voice :", VOICE_MAPPING.get(cl.user_session.get("Language")))
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    client_contexts["speech_synthesizer"] = speech_synthesizer

    cl.user_session.set("track_id", str(uuid4()))
    # collected_messages = []
    async def handle_conversation_updated(event):
        item = event.get("item")
        delta = event.get("delta")
        # logger.info("Conversation updated initiated")
        """Currently used to stream audio back to the client."""
        if delta:
            # Only one of the following will be populated for any given event
            if 'audio' in delta:
                audio = delta['audio']  # Int16Array, audio added
                if not cl.user_session.get("useAzureVoice"):
                    await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id")))
            if 'transcript' in delta:
                if cl.user_session.get("useAzureVoice"):
                    chunk_message = delta['transcript']
                    logger.info(f"Item status: {item['status']}, Chunk: {chunk_message}")
                    if item["status"] == "in_progress":
                        collected_messages.append(chunk_message)  # save the message
                        if chunk_message in tts_sentence_end: # sentence end found
                            sent_transcript = ''.join(collected_messages).strip()
                            collected_messages.clear()
                            if interrupt_audio and item["id"] != item_id:
                                pass
                            else:
                                chunk = await text_to_speech_realtime(speech_synthesizer=speech_synthesizer, text=sent_transcript, voice= voice)
                                await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="audio/wav", data=chunk, track=cl.user_session.get("track_id")))
            if 'arguments' in delta:
                arguments = delta['arguments']  # string, function arguments added
                pass
    
    async def handle_item_completed(item):
        """Generate the transcript once an item is completed and populate the chat context."""
        global item_id
        global interrupt_audio
        try:
            item_id = item["item"]["id"]
            interrupt_audio = False
            transcript = item['item']['formatted']['transcript']
            if transcript.strip() != "":
                await cl.Message(content=transcript).send()      
                
        except Exception as e:
            logger.error(f"Failed to generate transcript: {e}")
            logger.error(traceback.format_exc())
    
    async def handle_conversation_interrupt(event):
        """Used to cancel the client previous audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        logger.info("Interrupting Azure Speech TTS")
        try:
            cl.user_session.get("speech_synthesizer").stop_speaking_async()
            collected_messages.clear()
        except Exception as e:
            logger.error(f"Failed to clear collected messages: {e}")   
        await cl.context.emitter.send_audio_interrupt()

    async def handle_conversation_interrupt_azure_sdk(event):
        """Used to cancel the client previous audio playback."""
        global interrupt_audio
        global collected_messages
        cl.user_session.set("track_id", str(uuid4()))

        logger.info("Interrupting Azure Speech TTS")

        try:
            speech_synthesizer = client_contexts.get("speech_synthesizer")
            # speech_synthesizer.stop_speaking()
            speech_synthesizer.stop_speaking_async().get()
        # try:
        #     connection = speechsdk.Connection.from_speech_synthesizer(cl.user_session.get("speech_synthesizer"))
        #     connection.send_message_async('synthesis.control', '{"action":"stop"}').get()
            collected_messages.clear()
            logger.info(f"Collect messages cleared: {collected_messages}")
            await cl.user_session.get("openai_realtime").cancel_response()
            interrupt_audio = True
            logger.info("Interrupted Azure Speech TTS")
        except Exception as e:
            # logger.info("Sending messages through connection object is not yet supported by current Speech SDK")
            logger.error(f"Failed to clear collected messages: {e}")
        
    async def handle_input_audio_transcription_completed(event):
        item = event.get("item")
        delta = event.get("delta")
        if 'transcript' in delta:
            transcript = delta['transcript']
            if transcript != "":
                await cl.Message(author="You", type="user_message", content=transcript).send()
        
    async def handle_error(event):
        logger.error(event)
        
    
    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('conversation.item.input_audio_transcription.completed', handle_input_audio_transcription_completed)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    cl.user_session.set("speech_synthesizer", speech_synthesizer)
    #cl.user_session.set("tts_client", tts_client)
    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]
    await asyncio.gather(*coros)
    

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("raj", "pass123"):
        return cl.User(
            identifier="raj", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def start():
    logger.info("Chat started: Initializing settings")
    app_user = cl.user_session.get("user")
    logger.info(f"User: {app_user}")
    settings = {"useAzureVoice" : True, "Language" : "english", "Temperature" : 1}
    # settings = await cl.ChatSettings([
    #     Select(
    #         id="Language",
    #         label="Choose Language",
    #         values=list(VOICE_MAPPING.keys()),
    #         initial_index=0,
    #     ),
    #     # Switch(id="Language", label="Preferred Language", )
    #     Switch(id="useAzureVoice", label="Use Azure Voice", initial=True),
    #     Slider(
    #         id="Temperature",
    #         label="Temperature",
    #         initial=1,
    #         min=0,
    #         max=2,
    #         step=0.1,
    #     )
    # ]).send()
    logger.info(f"Settings sent: {settings}")
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    system_prompt = """You're a female customer support voice bot . Be consise in your response and speak in <customer_language> language always. """    
    
    cl.user_session.set("useAzureVoice", settings["useAzureVoice"])
    cl.user_session.set("Temperature", settings["Temperature"])
    cl.user_session.set("Language", settings["Language"])
    
    # Add logging to verify settings
    logger.info(f"Settings updated: useAzureVoice={settings['useAzureVoice']}, Language={settings['Language']}, Temperature={settings['Temperature']}")
    
    app_user = cl.user_session.get("user")
    identifier = app_user.identifier if app_user else "admin"
    await cl.Message(
        content="Hi, Welcome to ShopMe. How can I help you?. Press `P` to talk!"
    ).send()
    system_prompt = system_prompt.replace("<customer_language>", settings["Language"])
    await setup_openai_realtime(system_prompt=system_prompt + "\n\n Customer ID: 12121")
    
@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.send_user_message_content([{ "type": 'input_text', "text": message.content}])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()

@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        # TODO: might want to recreate items to restore context
        # openai_realtime.create_conversation_item(item)
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime:            
        if openai_realtime.is_connected():
            await openai_realtime.append_input_audio(chunk.data)
        else:
            logger.info("RealtimeClient is not connected")

@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()
