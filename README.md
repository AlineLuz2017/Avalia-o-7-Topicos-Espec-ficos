# Avalia-o-7-Topicos-Espec-ficos
#  Transcrição de Áudio com Whisper e Extração de Pontos-Chave com Ollama

Pipeline automatizado desenvolvido para a **Avaliação Online**, utilizando o modelo **Whisper** da OpenAI para transcrição de áudio e o **Ollama (Llama 3)** para a extração de resumos e pontos-chave estruturados em ambiente **Google Colab**.

---

##  Descrição da Atividade

O projeto executa um fluxo de trabalho dividido em quatro etapas principais:
1. **Passo 1 (Instalação):** Configuração do ambiente, instalando o gerenciador `zstd`, as bibliotecas Python do Whisper, o Ollama e o binário do serviço.
2. **Passo 2 (Inicialização):** Inicialização do servidor Ollama em segundo plano e download do modelo `llama3`.
3. **Passo 3 (Geração de Áudio):** Criação automatizada de um arquivo de áudio de teste (`teste_audio.mp3`) utilizando Text-to-Speech (`gTTS`) com foco no tema *Saúde Mental e Tecnologia*.
4. **Passo 4 (Pipeline de Execução):** Transcrição do áudio via Whisper e envio do texto transcrito para o Ollama gerar uma visão geral e pontos-chave estruturados.

---

##  Instruções para Instalação e Execução

### Célula 1: Instalando zstd, Whisper e dependências
```python
print("🔧 Instalando zstd (necessário para o Ollama)...")
!apt-get update && apt-get install -y zstd -qq

print("🔧 Instalando Whisper e dependências...")
!pip install -U openai-whisper git+[https://github.com/openai/whisper.git](https://github.com/openai/whisper.git) -q
!pip install ollama -q

print("🔧 Instalando o binário do Ollama...")

Célula 2: Inicializando o Ollama e baixando o modelo

import subprocess
import time

print("🚀 Iniciando o servidor do Ollama em segundo plano...")
ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(6) # Aguarda o servidor estabilizar

print("📥 Baixando o modelo llama3 (isso pode levar de 30 a 60 segundos)...")
!ollama pull llama3

print("✅ Servidor Ollama pronto e modelo baixado!")

Célula 3: Gerando o áudio de teste sobre Saúde Mental e Tecnologia

!pip install gtts -q
from gtts import gTTS

texto_saude_mental = """
Olá! Sejam bem-vindos a esta discussão sobre o impacto da tecnologia na nossa saúde mental.
Hoje vamos destacar três pilares fundamentais.
Primeiro, o uso excessivo de telas, especialmente antes de dormir, interfere diretamente na produção de melatonina, prejudicando a qualidade do sono e gerando fadiga crônica.
Segundo, fenômenos como o FOMO, que é o medo de estar perdendo algo, aumentam a ansiedade devido à comparação social constante nas redes sociais.
E terceiro, a importância do detox digital e de pausas programadas longe dos dispositivos para recuperar o foco e promover o bem-estar emocional.
"""

tts = gTTS(text=texto_saude_mental, lang='pt')
tts.save("teste_audio.mp3")

print("✅ Áudio de 'Saúde Mental e Tecnologia' gerado com sucesso como 'teste_audio.mp3'!")

Célula 4: Transcrição e Extração de Pontos-Chave
import whisper
import ollama
import os

print("🎧 Carregando o modelo Whisper (base)...")
whisper_model = whisper.load_model("base")

# Escolha o arquivo de áudio:
# Se usou a Célula 3, deixe "teste_audio.mp3".
# Se enviou seu próprio arquivo, coloque o nome exato dele aqui (ex: "minha_gravacao.mp3")
caminho_audio = "teste_audio.mp3"

if not os.path.exists(caminho_audio):
    print(f"❌ Erro: O arquivo '{caminho_audio}' não foi encontrado. Faça o upload dele na barra lateral esquerda do Colab.")
else:
    print(f"\n[1/2] 🎙️ Transcrevendo o áudio '{caminho_audio}'...")
    resultado = whisper_model.transcribe(caminho_audio, language="pt")
    texto_transcrito = resultado["text"]

    print("\n--- Texto Transcrito ---")
    print(texto_transcrito)

    print("\n[2/2] 🤖 Processando o texto com o Ollama para extrair os pontos-chave...")
    prompt = f"""
    Analise o texto transcrito abaixo e crie um resumo estruturado contendo:
    1. Uma visão geral concisa do áudio.
    2. Os principais pontos-chave listados em bullet points (tópicos).

    Texto Transcrito:
    \"\"\"{texto_transcrito}\"\"\"
    """

    resposta = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    resumo_final = resposta['message']['content']

    print("\n" + "="*50)
    print("📋 RESULTADO FINAL: RESUMO E PONTOS-CHAVE")
    print("="*50)
    print(resumo_final)
    print("="*50)

    Exemplo de Saída Obtida no Console
    --- Texto Transcrito ---
Olá, sejam bem-vindos a esta discussão sobre o impacto da tecnologia na nossa saúde mental. Hoje vamos destacar três pilares fundamentais. Primeiro, o uso excessivo de telas, especialmente antes de dormir, interfere diretamente na produção de melatonina, prejudicando a qualidade do sono e gerando fadiga crônica. Segundo, fenômenos como o fomo, que é o medo de estar perdendo algo. Almentam a ansiedade devido a comparação social constante nas redes sociais. E terceiro, a importância do detox digital e de pausas programadas longe dos dispositivos para recuperar o foco. E promover o bem-estar emocional.

==================================================
📋 RESULTADO FINAL: RESUMO E PONTOS-CHAVE
==================================================
**Visão Geral:**
O áudio discute o impacto da tecnologia na saúde mental, destacando três pilares fundamentais que afetam a qualidade do sono, a ansiedade e o bem-estar emocional.

**Principais Pontos-Chave:**
- **Uso excessivo de telas:** A exposição excessiva a telas, especialmente antes de dormir, interfere na produção de melatonina, prejudicando o sono e gerando fadiga crônica.
- **Fomo (medo de perder algo):** O fenômeno do fomo, alimentado pelas redes sociais, aumenta a ansiedade devido à comparação social constante.
- **Detox digital e pausas programadas:** A importância de pausar longe dos dispositivos para recuperar o foco e promover o bem-estar emocional.
- **Impacto na saúde mental:** O áudio destaca o impacto negativo da tecnologia na saúde mental, enfatizando a necessidade de equilibrar a vida digital com a vida emocional e física.
==================================================

Célula 2: Inicializando o Ollama e baixando o modelo!curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

print("✅ Dependências instaladas com sucesso!")
