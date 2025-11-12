"""Text-to-Speech engine with offline and online modes."""

import os
import tempfile
from typing import Optional


class TTSEngine:
    """Handles text-to-speech conversion with multiple backends."""
    
    def __init__(self, mode="offline", rate=150, volume=1.0, language="en"):
        """
        Initialize TTS engine.
        
        Args:
            mode: "offline" for pyttsx3 or "online" for gTTS
            rate: Speech rate (words per minute) for offline mode
            volume: Volume level (0.0 to 1.0) for offline mode
            language: Language code for TTS
        """
        self.mode = mode
        self.rate = rate
        self.volume = volume
        self.language = language
        self.engine = None
        
        if mode == "offline":
            self._init_offline()
        
    def _init_offline(self):
        """Initialize offline TTS engine (pyttsx3)."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
        except Exception as e:
            print(f"Warning: Could not initialize offline TTS: {e}")
            self.engine = None
    
    def speak(self, text: str):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
        """
        if not text:
            return
        
        try:
            if self.mode == "offline":
                self._speak_offline(text)
            else:
                self._speak_online(text)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def _speak_offline(self, text: str):
        """Speak using offline TTS (pyttsx3)."""
        if self.engine is not None:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            print(f"TTS (offline not available): {text}")
    
    def _speak_online(self, text: str):
        """Speak using online TTS (gTTS)."""
        try:
            from gtts import gTTS
            import pygame
            
            # Generate speech
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)
            
            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Cleanup
            pygame.mixer.quit()
            os.unlink(temp_file)
            
        except ImportError:
            print(f"TTS (online not available): {text}")
        except Exception as e:
            print(f"TTS online error: {e}")
    
    def stop(self):
        """Stop any ongoing speech."""
        if self.mode == "offline" and self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
