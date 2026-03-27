import os
import asyncio
import edge_tts
import pygame
import threading
import uuid

# 初始化音频播放器
pygame.mixer.init()

# 播放状态标志
_is_playing = False


def _play_audio(text):
    """内部函数：处理合成与播放"""
    global _is_playing

    async def amain():
        global _is_playing
        voice = "zh-CN-YunxiNeural"
        output_file = f"speech_{uuid.uuid4().hex}.mp3"

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)

            if not os.path.exists(output_file):
                print(f"[TTS] 错误：文件生成失败: {output_file}")
                return

            _is_playing = True
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

            pygame.mixer.music.unload()

        except Exception as e:
            print(f"[TTS] 出错: {e}")
        finally:
            _is_playing = False
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except Exception:
                    pass

    asyncio.run(amain())


def speak(text):
    """外部调用的接口，自动开启新线程，不卡主程序"""
    threading.Thread(target=_play_audio, args=(text,), daemon=True).start()


def is_playing():
    """返回当前是否正在播放语音"""
    return pygame.mixer.music.get_busy()


if __name__ == "__main__":
    print("--- Edge-TTS 高保真语音版 ---")
    print("输入文字按回车，输入 q 退出")

    while True:
        user_input = input("\n请输入内容: ")
        if user_input.lower() == 'q':
            break

        print(f"正在朗读: {user_input}...")
        speak(user_input)

    print("程序已退出")
