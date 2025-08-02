import speech_recognition as sr
import pyttsx3
import webbrowser
import wikipediaapi


engine = pyttsx3.init()

def speak(texto):
    print(f"Assistente: {texto}")
    engine.say(texto)
    engine.runAndWait()

def ouvir():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Diga algo...")
        audio = recognizer.listen(source)

    try:
        comando = recognizer.recognize_google(audio, language="pt-BR")
        print(f"Você disse: {comando}")
        return comando.lower()
    except sr.UnknownValueError:
        speak("Desculpe, não entendi.")
        return ""
    except sr.RequestError:
        speak("Erro ao conectar com o serviço.")
        return ""

def buscar_wikipedia(termo):
    wiki = wikipediaapi.Wikipedia('pt')
    page = wiki.page(termo)
    if page.exists():
        resumo = page.summary[:300]
        speak(f"Encontrei na Wikipedia: {resumo}")
    else:
        speak("Não encontrei nada na Wikipedia.")

def abrir_youtube():
    speak("Abrindo o YouTube.")
    webbrowser.open("https://www.youtube.com")

def abrir_farmacia():
    speak("Procurando farmácia próxima.")
    webbrowser.open("https://www.google.com/maps/search/farmácia+próxima/")

# Programa principal
def main():
    speak("Olá! Sou sua assistente virtual. Diga um comando.")
    while True:
        comando = ouvir()

        if "wikipedia" in comando:
            speak("O que você quer buscar na Wikipedia?")
            termo = ouvir()
            if termo:
                buscar_wikipedia(termo)

        elif "youtube" in comando:
            abrir_youtube()

        elif "farmácia" in comando or "farmacia" in comando:
            abrir_farmacia()

        elif "sair" in comando or "tchau" in comando:
            speak("Até mais!")
            break

        elif comando:
            speak("Comando não reconhecido. Tente de novo.")

if __name__ == "__main__":
    main()
