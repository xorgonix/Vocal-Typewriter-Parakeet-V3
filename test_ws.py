import asyncio
import websockets
import json
import numpy as np
import subprocess
import os

async def test_transcription():
    uri = "ws://localhost:8000/ws"
    wav_path = "models/parakeet-tdt-0.6b-v3-int8/test_wavs/en.wav"
    
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return

    print(f"Converting {wav_path} to 16kHz float32...")
    cmd = [
        "ffmpeg", 
        "-i", wav_path,
        "-f", "f32le", 
        "-acodec", "pcm_f32le", 
        "-ac", "1", 
        "-ar", "16000", 
        "-"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    
    print(f"Got {len(audio_data)} bytes of audio data")
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Send in chunks
        chunk_size = 4096 * 4 # 4 bytes per float
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await websocket.send(chunk)
            await asyncio.sleep(0.01) # Simulate real-time slightly
            
            # Try to receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                data = json.loads(response)
                print(f"Received: {data}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"Error receiving: {e}")
                
        # Send some silence at the end to flush VAD
        silence = np.zeros(16000 * 3, dtype=np.float32).tobytes() # 3 seconds
        # Send silence in chunks too
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i+chunk_size]
            await websocket.send(chunk)
            await asyncio.sleep(0.01)
            
            # Try to receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                data = json.loads(response)
                print(f"Received during silence: {data}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"Error receiving during silence: {e}")
        
        # Wait for final results
        print("Waiting for final results...")
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                print(f"Final Receive: {response}")
        except asyncio.TimeoutError:
            print("Finished waiting")

if __name__ == "__main__":
    asyncio.run(test_transcription())
