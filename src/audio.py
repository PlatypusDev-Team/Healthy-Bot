import os
import tempfile

from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

def transcrever_audio(bot, message):
    caminho_audio = os.path.join(
        tempfile.gettempdir(),
        f"audio_{message.chat.id}.ogg"
    )

    try:
        bot.send_message(
            message.chat.id,
            "🎤 Áudio recebido, transcrevendo..."
        )

        # Baixa o arquivo
        file_info = bot.get_file(message.voice.file_id)

        arquivo_baixado = bot.download_file(
            file_info.file_path
        )

        with open(caminho_audio, "wb") as f:
            f.write(arquivo_baixado)

        # Transcreve
        segments, info = model.transcribe(
            caminho_audio,
            language="pt"
        )

        texto = "".join(
            segment.text
            for segment in segments
        ).strip()

        if not texto:
            texto = "Não consegui entender o áudio. Pode repetir, por favor??"

        return texto

    except Exception as erro:
        print(f"ERRO ÁUDIO: {erro}")
        return None

    finally:
        if os.path.exists(caminho_audio):
            os.remove(caminho_audio)