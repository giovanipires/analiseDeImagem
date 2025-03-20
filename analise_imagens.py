import cv2
import tensorflow as tf
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Frame, Text, messagebox
from tkinter import ttk  # Para a barra de progresso
from PIL import Image, ImageTk
import threading
import time

# Variável global para controle de cancelamento
processamento_cancelado = False

# Função para carregar a imagem
def carregar_imagem():
    # Abre uma janela para o usuário selecionar uma imagem
    caminho_imagem = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Todos os arquivos", "*.*")]  # Mostra todos os arquivos
    )
    if not caminho_imagem:
        messagebox.showinfo("Aviso", "Nenhuma imagem selecionada. Tente novamente.")
        return None
    return caminho_imagem

# Função para pré-processar a imagem para o modelo
def preprocessar_imagem(imagem):
    imagem = cv2.resize(imagem, (224, 224))  # Redimensiona para o tamanho esperado pelo MobileNet
    imagem = tf.keras.applications.mobilenet.preprocess_input(imagem)  # Pré-processamento específico
    return np.expand_dims(imagem, axis=0)  # Adiciona uma dimensão de batch

# Função para analisar a imagem
def analisar_imagem(imagem):
    # Carrega o modelo MobileNet pré-treinado
    modelo = tf.keras.applications.MobileNet(weights="imagenet")
    previsoes = modelo.predict(imagem)
    resultados = tf.keras.applications.mobilenet.decode_predictions(previsoes, top=5)[0]  # Top 5 resultados
    return resultados

# Função para exibir a imagem na interface
def exibir_imagem(caminho_imagem):
    # Abre a imagem com Pillow
    imagem_pil = Image.open(caminho_imagem)
    imagem_pil = imagem_pil.resize((400, 400))  # Redimensiona para caber na interface
    imagem_tk = ImageTk.PhotoImage(imagem_pil)

    # Exibe a imagem no label
    label_imagem.config(image=imagem_tk)
    label_imagem.image = imagem_tk  # Mantém uma referência para evitar garbage collection

    # Retorna a imagem para análise
    return cv2.imread(caminho_imagem)

# Função para cancelar o processamento
def cancelar_processamento():
    global processamento_cancelado
    processamento_cancelado = True
    btn_cancelar.config(state="disabled")  # Desabilita o botão de cancelamento após clicar

# Função para iniciar a análise em uma thread separada
def iniciar_analise():
    global processamento_cancelado
    processamento_cancelado = False  # Reseta a variável de cancelamento

    caminho_imagem = carregar_imagem()
    if caminho_imagem:
        # Configura a interface para o processamento
        texto_resultados.pack(pady=10)  # Exibe a área de resultados
        texto_resultados.delete(1.0, "end")  # Limpa o conteúdo anterior
        barra_progresso.pack(pady=10)  # Exibe a barra de progresso
        barra_progresso["value"] = 0  # Reseta a barra de progresso
        btn_selecionar.config(state="disabled")  # Desabilita o botão de seleção
        btn_cancelar.config(state="normal")  # Habilita o botão de cancelamento
        label_status.config(text="Processando...", fg="blue")  # Atualiza o status

        # Executa a análise em uma thread separada para não travar a interface
        def processar():
            try:
                # Exibe a imagem
                imagem = exibir_imagem(caminho_imagem)
                if imagem is not None:
                    # Simula o progresso de pré-processamento
                    for i in range(25):
                        if processamento_cancelado:
                            break
                        time.sleep(0.1)  # Simula um delay
                        barra_progresso["value"] += 1
                        root.update()  # Atualiza a interface

                    # Pré-processa a imagem
                    if not processamento_cancelado:
                        imagem_processada = preprocessar_imagem(imagem)

                        # Simula o progresso de análise
                        for i in range(25, 100):
                            if processamento_cancelado:
                                break
                            time.sleep(0.1)  # Simula um delay
                            barra_progresso["value"] += 1
                            root.update()  # Atualiza a interface

                        # Analisa a imagem
                        if not processamento_cancelado:
                            resultados = analisar_imagem(imagem_processada)

                            # Exibe os resultados na interface
                            texto_resultados.insert("end", "Resultados da análise:\n\n")
                            for i, (id, label, probabilidade) in enumerate(resultados):
                                texto_resultados.insert("end", f"{i + 1}. {label} ({probabilidade * 100:.2f}%)\n")

                            # Atualiza o status
                            label_status.config(text="Análise concluída!", fg="green")
            except Exception as e:
                label_status.config(text="Erro ao processar a imagem.", fg="red")
                messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
            finally:
                # Restaura a interface
                btn_selecionar.config(state="normal")
                btn_cancelar.config(state="disabled")
                if processamento_cancelado:
                    label_status.config(text="Processamento cancelado.", fg="orange")
                    barra_progresso["value"] = 0  # Reseta a barra de progresso

        # Inicia a thread de processamento
        threading.Thread(target=processar).start()

# Configuração da interface gráfica
root = Tk()
root.title("Análise de Imagens")

# Frame principal
frame_principal = Frame(root)
frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

# Label para exibir a imagem
label_imagem = Label(frame_principal)
label_imagem.pack(pady=10)

# Botão para selecionar a imagem
btn_selecionar = Button(frame_principal, text="Selecionar Imagem", command=iniciar_analise, font=("Arial", 12))
btn_selecionar.pack(pady=10)

# Botão para cancelar o processamento
btn_cancelar = Button(frame_principal, text="Cancelar", command=cancelar_processamento, font=("Arial", 12), state="disabled")
btn_cancelar.pack(pady=10)

# Barra de progresso (inicialmente oculta)
barra_progresso = ttk.Progressbar(frame_principal, orient="horizontal", length=300, mode="determinate")
barra_progresso.pack_forget()  # Oculta a barra de progresso inicialmente

# Área para exibir os resultados (inicialmente oculta)
texto_resultados = Text(frame_principal, width=50, height=10, font=("Arial", 12))
texto_resultados.pack_forget()  # Oculta a área de resultados inicialmente

# Label para o status do processamento
label_status = Label(frame_principal, text="", font=("Arial", 12))
label_status.pack(pady=10)

# Inicia a interface gráfica
root.mainloop()
